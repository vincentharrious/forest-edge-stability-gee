#!/usr/bin/env python3
"""Submit four dense annual-NIRv canaries for the spatial-structure pilot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd
from google.cloud import storage
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[2]
CANARY_INPUT = (
    ROOT / "analysis_inputs" / "spatial_structure_pilot" / "canary_patches.csv"
)
SELECTED_INPUT = (
    ROOT / "analysis_inputs" / "spatial_structure_pilot" / "selected_patches.csv"
)
SAMPLE_INPUT = (
    ROOT / "analysis_inputs" / "annual_climate_state" / "formal_sample.parquet"
)
PATCH_INPUT = (
    ROOT / "analysis_inputs" / "patch_context" / "focal_patch_identity.parquet"
)
TASK_REGISTRY = (
    ROOT
    / "outputs"
    / "exploratory"
    / "dense_nirv_spatial_pilot"
    / "canary_task_registry_project_5bef1c0a_v3.json"
)
FROZEN_PREFLIGHT_REGISTRY = (
    ROOT
    / "outputs"
    / "exploratory"
    / "dense_nirv_spatial_pilot"
    / "canary_task_registry_project_5bef1c0a.json"
)

PROJECT = "project-5bef1c0a-62c5-45b8-8c4"
BUCKET = "temporal-stability-5bef1c0a-work"
OUTPUT_PREFIX = (
    "regional/northeast/v1-32-dense-nirv-spatial-pilot/"
    "canary-v3-project-5bef1c0a"
)
DESCRIPTION_PREFIX = "v132-dense-nirv-spatial-pilot-canary-v3-p5bef1c0a"
FOREST_LATTICE_ASSET = (
    "projects/blissful-potion-503913-c0/assets/sf0c/"
    "northeast_v132_forest_lattice_full_v1"
)
DISTANCE_LATTICE_ASSET = (
    "projects/blissful-potion-503913-c0/assets/sf0c/"
    "northeast_v132_distance_lattice_full_v1"
)
L30 = "NASA/HLS/HLSL30/v002"

YEARS = tuple(range(2013, 2026))
MONTHS = (6, 7, 8)
SCALE_M = 30
MINIMUM_VALID_YEAR_N = 10
MINIMUM_DISTANCE_M = 30
MAXIMUM_DISTANCE_M = 480
COMPONENT_MAX_SIZE = 1024
COMPONENT_REGION_RADIUS_M = 25_000
LATTICE_TRANSFORM = (30, 0, 11_040_000, 0, -30, 5_997_270)

TO_WGS84 = Transformer.from_crs(6933, 4326, always_xy=True)
FIXED_BANDS = (
    "baseline_patch_root",
    "lattice_col",
    "lattice_row",
    "valid_year_n",
    "d_between_m",
)
RESPONSE_BANDS = tuple(
    name
    for year in YEARS
    for name in (
        f"valid_scene_n_{year}",
        f"distance_m_{year}",
        f"nirv_{year}",
    )
)
EXPORT_SELECTORS = FIXED_BANDS + RESPONSE_BANDS
EXTRA_COMPONENT_SEEDS = {
    2_059_916: ((12_693, 19_288),),
}


def _projection() -> ee.Projection:
    return ee.Image(FOREST_LATTICE_ASSET).select(0).projection()


def _lattice_center(row: int, col: int) -> tuple[float, float]:
    x = LATTICE_TRANSFORM[2] + (col + 0.5) * LATTICE_TRANSFORM[0]
    y = LATTICE_TRANSFORM[5] + (row + 0.5) * LATTICE_TRANSFORM[4]
    return x, y


def _point_and_region(row: int, col: int) -> tuple[ee.Geometry, ee.Geometry]:
    x, y = _lattice_center(row, col)
    lon, lat = TO_WGS84.transform(x, y)
    region_wgs84 = transform(
        TO_WGS84.transform,
        box(
            x - COMPONENT_REGION_RADIUS_M,
            y - COMPONENT_REGION_RADIUS_M,
            x + COMPONENT_REGION_RADIUS_M,
            y + COMPONENT_REGION_RADIUS_M,
        ),
    )
    point = ee.Geometry.Point([lon, lat])
    region = ee.Geometry(mapping(region_wgs84), proj="EPSG:4326", geodesic=False)
    return point, region


def _qa_valid(image: ee.Image) -> ee.Image:
    fmask = image.select("Fmask").unmask(255)
    invalid_surface_bits = fmask.bitwiseAnd(62).neq(0)
    high_aerosol = fmask.rightShift(6).bitwiseAnd(3).eq(3)
    band_valid = image.select(["B4", "B5"]).mask().reduce(ee.Reducer.min())
    return (
        fmask.neq(255)
        .And(invalid_surface_bits.Not())
        .And(high_aerosol.Not())
        .And(band_valid)
        .rename("qa_valid")
    )


def _with_qa_valid(image: ee.Image) -> ee.Image:
    return image.addBands(_qa_valid(image))


def _nirv(image: ee.Image) -> ee.Image:
    red = image.select("B4")
    nir = image.select("B5")
    ndvi = nir.subtract(red).divide(nir.add(red))
    return ndvi.multiply(nir).rename("nirv").updateMask(image.select("qa_valid"))


def _annual_scene_count(collection: ee.ImageCollection, year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, MONTHS[0], 1)
    end = ee.Date.fromYMD(year, MONTHS[-1] + 1, 1)
    return (
        collection.filterDate(start, end)
        .select("qa_valid")
        .map(lambda image: image.unmask(0).uint8())
        .sum()
        .rename(f"valid_scene_n_{year}")
        .uint16()
        .unmask(0)
    )


def _annual_nirv(collection: ee.ImageCollection, year: int) -> ee.Image:
    start = ee.Date.fromYMD(year, MONTHS[0], 1)
    end = ee.Date.fromYMD(year, MONTHS[-1] + 1, 1)
    return (
        collection.filterDate(start, end)
        .map(_nirv)
        .median()
        .rename(f"nirv_{year}")
    )


def _attach_seed_pixels(selected: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_parquet(
        SAMPLE_INPUT,
        columns=("pixel_id", "lattice_row", "lattice_col"),
    )
    identity = pd.read_parquet(
        PATCH_INPUT,
        columns=("pixel_id", "baseline_patch_root"),
    )
    seeds = (
        sample.merge(identity, on="pixel_id")
        .sort_values("pixel_id")
        .drop_duplicates("baseline_patch_root")
        .rename(columns={"baseline_patch_root": "patch"})
    )
    selected = selected.merge(
        seeds[["patch", "pixel_id", "lattice_row", "lattice_col"]],
        on="patch",
        how="left",
        validate="one_to_one",
    )
    if selected[["lattice_row", "lattice_col"]].isna().any().any():
        raise RuntimeError("selected patches do not all have unique seed pixels")
    selected["lattice_row"] = selected["lattice_row"].astype(int)
    selected["lattice_col"] = selected["lattice_col"].astype(int)
    return selected.sort_values(["spatial_block", "patch"]).reset_index(drop=True)


def _load_canaries() -> pd.DataFrame:
    canaries = _attach_seed_pixels(pd.read_csv(CANARY_INPUT))
    if len(canaries) != 4:
        raise RuntimeError("the frozen canary set does not contain four patches")
    return canaries


def load_selected_patches() -> pd.DataFrame:
    selected = _attach_seed_pixels(pd.read_csv(SELECTED_INPUT))
    if len(selected) != 48 or selected["patch"].duplicated().any():
        raise RuntimeError("the frozen dense pilot does not contain 48 unique patches")
    return selected


def _component_from_seed(row: int, col: int) -> tuple[ee.Image, ee.Geometry]:
    point, region = _point_and_region(row, col)
    baseline = (
        ee.Image(FOREST_LATTICE_ASSET)
        .select("baseline_forest_2013")
        .selfMask()
        .clip(region)
    )
    labels = baseline.connectedComponents(
        ee.Kernel.square(1), COMPONENT_MAX_SIZE
    ).select("labels")
    label = ee.Number(
        labels.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            crs=_projection(),
            maxPixels=10_000,
        ).get("labels")
    )
    return labels.eq(label).selfMask(), region


def _patch_mask(row: pd.Series) -> tuple[ee.Image, ee.Geometry]:
    patch = int(row.patch)
    seeds = (
        (int(row.lattice_row), int(row.lattice_col)),
        *EXTRA_COMPONENT_SEEDS.get(patch, ()),
    )
    masks_and_regions = [
        _component_from_seed(seed_row, seed_col) for seed_row, seed_col in seeds
    ]
    mask, region = masks_and_regions[0]
    for extra_mask, extra_region in masks_and_regions[1:]:
        mask = mask.unmask(0, False).Or(extra_mask.unmask(0, False)).selfMask()
        region = region.union(extra_region, maxError=1)
    return mask.rename("patch_mask"), region


def _annual_inputs(
    region: ee.Geometry,
) -> tuple[ee.ImageCollection, ee.Image, ee.Image]:
    forest_lattice = ee.Image(FOREST_LATTICE_ASSET)
    baseline = forest_lattice.select("baseline_forest_2013")
    loss = forest_lattice.select("lossyear_code").unmask(0)
    collection = (
        ee.ImageCollection(L30)
        .filterBounds(region)
        .filterDate(
            ee.Date.fromYMD(YEARS[0], MONTHS[0], 1),
            ee.Date.fromYMD(YEARS[-1], MONTHS[-1] + 1, 1),
        )
        .filter(ee.Filter.calendarRange(MONTHS[0], MONTHS[-1], "month"))
        .map(_with_qa_valid)
    )
    return collection, baseline, loss


def _patch_pixel_images(
    row: pd.Series,
) -> tuple[ee.Image, ee.Image, ee.Geometry]:
    patch_mask, region = _patch_mask(row)
    collection, baseline, loss = _annual_inputs(region)
    valid_year_n = None
    log_distance_sum = None
    annual_bands: list[ee.Image] = []
    for year in YEARS:
        forest = baseline.And(
            loss.gt(12).And(loss.add(2000).lte(year - 1)).Not()
        )
        distance = (
            ee.Image(DISTANCE_LATTICE_ASSET)
            .select(f"distance_sq_px_{year}")
            .sqrt()
            .multiply(SCALE_M)
            .rename(f"distance_m_{year}")
        )
        scene_count = _annual_scene_count(collection, year)
        valid = forest.And(distance.lt(100_000)).And(scene_count.gte(1))
        valid_increment = valid.unmask(0).uint8()
        log_distance_increment = distance.log10().updateMask(valid).unmask(0)
        if valid_year_n is None:
            valid_year_n = valid_increment
            log_distance_sum = log_distance_increment
        else:
            valid_year_n = valid_year_n.add(valid_increment)
            log_distance_sum = log_distance_sum.add(log_distance_increment)
        annual_bands.extend(
            (
                scene_count,
                distance.updateMask(valid),
                _annual_nirv(collection, year).updateMask(valid),
            )
        )
    valid_year_n = valid_year_n.rename("valid_year_n").uint8()
    d_between = (
        ee.Image.constant(10)
        .pow(log_distance_sum.divide(valid_year_n))
        .rename("d_between_m")
    )
    eligible = (
        patch_mask
        .And(valid_year_n.gte(MINIMUM_VALID_YEAR_N))
        .And(d_between.gt(MINIMUM_DISTANCE_M))
        .And(d_between.lte(MAXIMUM_DISTANCE_M))
    )
    coordinates = (
        ee.Image.pixelCoordinates(_projection())
        .floor()
        .toInt64()
        .rename(["lattice_col", "lattice_row"])
    )
    fixed = (
        ee.Image.constant(int(row.patch))
        .toInt64()
        .rename("baseline_patch_root")
        .addBands(coordinates)
        .addBands(valid_year_n)
        .addBands(d_between)
    )
    response = annual_bands[0]
    for band in annual_bands[1:]:
        response = response.addBands(band)
    return fixed.updateMask(eligible), response.updateMask(eligible), region


def build_export_collection(row: pd.Series) -> ee.FeatureCollection:
    fixed_image, response_image, region = _patch_pixel_images(row)
    fixed_pixels = fixed_image.sample(
        region=region,
        projection=_projection(),
        dropNulls=True,
        tileScale=4,
        geometries=True,
    )
    return response_image.reduceRegions(
        collection=fixed_pixels,
        reducer=ee.Reducer.first(),
        crs=_projection(),
        tileScale=4,
    )


def _frozen_preflight() -> dict:
    source = json.loads(FROZEN_PREFLIGHT_REGISTRY.read_text(encoding="utf-8"))
    preflight = source["preflight"]
    canaries = _load_canaries()
    expected_patches = set(canaries["patch"].astype(int))
    observed_patches = {int(row["patch"]) for row in preflight["patches"]}
    if (
        preflight["status"] != "CANARY_PREFLIGHT_PASS"
        or preflight["canary_patch_n"] != 4
        or preflight["eligible_pixel_n"] != 52_361
        or observed_patches != expected_patches
        or any(
            row["patch_pixel_n"] != row["expected_patch_pixel_n"]
            for row in preflight["patches"]
        )
    ):
        raise RuntimeError("frozen canary preflight is not the verified four-patch result")
    return {
        **preflight,
        "output_prefix": OUTPUT_PREFIX,
        "reused_without_hls_rescan": True,
        "source_registry": str(FROZEN_PREFLIGHT_REGISTRY.relative_to(ROOT)),
    }


def plan() -> dict:
    return _frozen_preflight()


def _submission_preflight() -> None:
    existing_tasks = [
        task
        for task in ee.data.getTaskList()
        if str(task.get("description", "")).startswith(DESCRIPTION_PREFIX)
    ]
    if existing_tasks:
        raise RuntimeError("canary tasks already exist; refusing duplicate submission")
    client = storage.Client(project=PROJECT)
    blobs = client.list_blobs(BUCKET, prefix=OUTPUT_PREFIX.rstrip("/") + "/")
    if next(iter(blobs), None) is not None:
        raise RuntimeError("canary output prefix is not empty; refusing overwrite")
    if TASK_REGISTRY.exists():
        raise RuntimeError("canary task registry already exists; refusing duplicate submission")


def submit() -> dict:
    ee.Initialize(project=PROJECT)
    preflight = plan()
    _submission_preflight()
    expected_by_patch = {
        int(row["patch"]): int(row["eligible_30_480_pixel_n"])
        for row in preflight["patches"]
    }
    registry = {
        "state": "SUBMITTING",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT,
        "bucket": BUCKET,
        "output_prefix": OUTPUT_PREFIX,
        "task_split": "one baseline forest patch per task; all 13 years wide",
        "export_selectors": list(EXPORT_SELECTORS),
        "preflight": preflight,
        "tasks": [],
    }
    TASK_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    canaries = _load_canaries()
    for _, row in canaries.iterrows():
        patch = int(row.patch)
        description = f"{DESCRIPTION_PREFIX}-{patch}"
        file_prefix = f"{OUTPUT_PREFIX}/{patch}"
        collection = build_export_collection(row)
        task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=description,
            bucket=BUCKET,
            fileNamePrefix=file_prefix,
            fileFormat="CSV",
            selectors=list(EXPORT_SELECTORS),
        )
        task.start()
        registry["tasks"].append(
            {
                "patch": patch,
                "spatial_block": str(row.spatial_block),
                "task_id": task.id,
                "description": description,
                "output_prefix": file_prefix,
                "expected_row_n": expected_by_patch[patch],
                "state_at_registration": "SUBMITTED",
            }
        )
        TASK_REGISTRY.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    registry["state"] = "SUBMITTED"
    registry["submission_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    TASK_REGISTRY.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "CANARIES_SUBMITTED",
        "task_n": len(registry["tasks"]),
        "eligible_pixel_n": preflight["eligible_pixel_n"],
        "task_registry": str(TASK_REGISTRY.relative_to(ROOT)),
        "output_prefix": OUTPUT_PREFIX,
    }


def status() -> dict:
    ee.Initialize(project=PROJECT)
    registry = json.loads(TASK_REGISTRY.read_text(encoding="utf-8"))
    task_ids = [task["task_id"] for task in registry["tasks"]]
    statuses = ee.data.getTaskStatus(task_ids)
    return {"task_n": len(statuses), "tasks": statuses}


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
