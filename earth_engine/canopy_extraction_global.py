#!/usr/bin/env python3
"""Submit the remaining 44 dense annual-NIRv patch exports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import ee
from google.cloud import storage

import submit_dense_nirv_spatial_pilot as pilot


ROOT = Path(__file__).resolve().parents[2]
COMPONENT_PREFLIGHT = (
    ROOT
    / "analysis_inputs"
    / "spatial_structure_pilot"
    / "full_component_preflight.json"
)
CANARY_REGISTRY = (
    ROOT
    / "outputs"
    / "exploratory"
    / "dense_nirv_spatial_pilot"
    / "canary_task_registry_project_5bef1c0a_v3.json"
)
TASK_REGISTRY = (
    ROOT
    / "outputs"
    / "exploratory"
    / "dense_nirv_spatial_pilot"
    / "full_task_registry_project_5bef1c0a_v1.json"
)

PROJECT = pilot.PROJECT
BUCKET = pilot.BUCKET
OUTPUT_PREFIX = (
    "regional/northeast/v1-32-dense-nirv-spatial-pilot/"
    "full-v1-project-5bef1c0a"
)
DESCRIPTION_PREFIX = "v132-dense-nirv-spatial-full-v1-p5bef1c0a"


def _load_scope():
    selected = pilot.load_selected_patches()
    canaries = pilot._load_canaries()
    canary_patches = set(canaries["patch"].astype(int))
    remaining = selected.loc[~selected["patch"].isin(canary_patches)].copy()
    if len(selected) != 48 or len(canaries) != 4 or len(remaining) != 44:
        raise RuntimeError("dense full scope is not 48 patches = 4 canaries + 44 remaining")
    return selected, canaries, remaining


def _local_preflight() -> dict:
    selected, canaries, remaining = _load_scope()
    component = json.loads(COMPONENT_PREFLIGHT.read_text(encoding="utf-8"))
    canary = json.loads(CANARY_REGISTRY.read_text(encoding="utf-8"))
    expected_extra_seeds = [list(seed) for seed in pilot.EXTRA_COMPONENT_SEEDS[2_059_916]]
    recorded_extra_seeds = component["multi_seed_patches"]["2059916"][
        "seeds_lattice_row_col"
    ][1:]
    if (
        component["status"] != "PASS"
        or component["selected_patch_n"] != 48
        or component["all_patch_pixel_counts_match_baseline_area"] is not True
        or component["selection_reads_nirv"] is not False
        or recorded_extra_seeds != expected_extra_seeds
    ):
        raise RuntimeError("full component preflight is not the verified 48-patch result")
    if (
        canary["state"] != "COMPLETED_VALIDATED"
        or canary["validation"]["completed_task_n"] != 4
        or canary["validation"]["exported_row_n"] != 52_361
        or canary["validation"]["column_n"] != len(pilot.EXPORT_SELECTORS)
    ):
        raise RuntimeError("four-patch canary is not the completed validated run")
    return {
        "status": "FULL_PREFLIGHT_PASS",
        "selected_patch_n": len(selected),
        "reused_canary_patch_n": len(canaries),
        "remaining_task_n": len(remaining),
        "estimated_total_eligible_pixel_n": float(
            selected["estimated_population_pixel_n_30_480"].sum()
        ),
        "estimated_remaining_eligible_pixel_n": float(
            remaining["estimated_population_pixel_n_30_480"].sum()
        ),
        "maximum_estimated_patch_row_n": float(
            remaining["estimated_population_pixel_n_30_480"].max()
        ),
        "component_preflight": str(COMPONENT_PREFLIGHT.relative_to(ROOT)),
        "canary_registry": str(CANARY_REGISTRY.relative_to(ROOT)),
        "output_prefix": OUTPUT_PREFIX,
    }


def plan() -> dict:
    return _local_preflight()


def _cloud_preflight(canary_registry: dict) -> None:
    existing_tasks = [
        task
        for task in ee.data.getTaskList()
        if str(task.get("description", "")).startswith(DESCRIPTION_PREFIX)
    ]
    if existing_tasks:
        raise RuntimeError("full-patch tasks already exist; refusing duplicate submission")
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    blobs = client.list_blobs(BUCKET, prefix=OUTPUT_PREFIX.rstrip("/") + "/")
    if next(iter(blobs), None) is not None:
        raise RuntimeError("full output prefix is not empty; refusing overwrite")
    for task in canary_registry["tasks"]:
        source_name = f'{canary_registry["output_prefix"]}/{int(task["patch"])}.csv'
        if not bucket.blob(source_name).exists(client):
            raise RuntimeError(f"validated canary output is missing: {source_name}")
    if TASK_REGISTRY.exists():
        raise RuntimeError("full task registry already exists; refusing duplicate submission")


def submit() -> dict:
    ee.Initialize(project=PROJECT)
    preflight = _local_preflight()
    selected, canaries, remaining = _load_scope()
    canary_registry = json.loads(CANARY_REGISTRY.read_text(encoding="utf-8"))
    _cloud_preflight(canary_registry)
    registry = {
        "state": "COPYING_VALIDATED_CANARIES",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT,
        "bucket": BUCKET,
        "output_prefix": OUTPUT_PREFIX,
        "task_split": "one baseline forest patch per task; all 13 years wide",
        "export_selectors": list(pilot.EXPORT_SELECTORS),
        "preflight": preflight,
        "reused_canaries": [],
        "tasks": [],
    }
    TASK_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    TASK_REGISTRY.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    for task in canary_registry["tasks"]:
        patch = int(task["patch"])
        source_name = f'{canary_registry["output_prefix"]}/{patch}.csv'
        destination_name = f"{OUTPUT_PREFIX}/{patch}.csv"
        copied = bucket.copy_blob(bucket.blob(source_name), bucket, destination_name)
        registry["reused_canaries"].append(
            {
                "patch": patch,
                "source": source_name,
                "destination": destination_name,
                "size_bytes": int(copied.size),
            }
        )
        TASK_REGISTRY.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    registry["state"] = "SUBMITTING_REMAINING_PATCHES"
    TASK_REGISTRY.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for index, (_, row) in enumerate(remaining.iterrows(), start=1):
        patch = int(row.patch)
        description = f"{DESCRIPTION_PREFIX}-{patch}"
        file_prefix = f"{OUTPUT_PREFIX}/{patch}"
        task = ee.batch.Export.table.toCloudStorage(
            collection=pilot.build_export_collection(row),
            description=description,
            bucket=BUCKET,
            fileNamePrefix=file_prefix,
            fileFormat="CSV",
            selectors=list(pilot.EXPORT_SELECTORS),
        )
        task.start()
        registry["tasks"].append(
            {
                "patch": patch,
                "spatial_block": str(row.spatial_block),
                "task_id": task.id,
                "description": description,
                "output_prefix": file_prefix,
                "estimated_row_n": float(
                    row.estimated_population_pixel_n_30_480
                ),
                "state_at_registration": "SUBMITTED",
            }
        )
        TASK_REGISTRY.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "submitted": index,
                    "total": len(remaining),
                    "patch": patch,
                    "task_id": task.id,
                }
            ),
            flush=True,
        )
    registry["state"] = "SUBMITTED"
    registry["submission_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    TASK_REGISTRY.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "FULL_48_PATCH_SCOPE_SUBMITTED",
        "reused_canary_patch_n": len(canaries),
        "submitted_task_n": len(registry["tasks"]),
        "total_patch_n": len(selected),
        "task_registry": str(TASK_REGISTRY.relative_to(ROOT)),
        "output_prefix": OUTPUT_PREFIX,
    }


def status() -> dict:
    ee.Initialize(project=PROJECT)
    registry = json.loads(TASK_REGISTRY.read_text(encoding="utf-8"))
    task_ids = [task["task_id"] for task in registry["tasks"]]
    statuses = ee.data.getTaskStatus(task_ids)
    counts: dict[str, int] = {}
    for task in statuses:
        state = str(task.get("state", "UNKNOWN"))
        counts[state] = counts.get(state, 0) + 1
    return {"task_n": len(statuses), "state_counts": counts, "tasks": statuses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "submit", "status"))
    args = parser.parse_args()
    if args.command == "plan":
        value = plan()
    elif args.command == "submit":
        value = submit()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
