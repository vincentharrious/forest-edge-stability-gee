#!/usr/bin/env python3
"""Recover 1991--2020 ERA5-Land JJA climate normals for the formal grids."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import CRS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_SAMPLE = ROOT / "analysis_inputs" / "annual_climate_state" / "formal_sample.parquet"
CURRENT_CLIMATE = (
    ROOT
    / "analysis_inputs"
    / "annual_climate_state"
    / "grid_year_climate_with_anomalies.parquet"
)
OUTPUT = (
    ROOT
    / "analysis_inputs"
    / "annual_climate_state"
    / "reference_1991_2020"
)
CHECKPOINTS = OUTPUT / "checkpoints"
YEARS = tuple(range(1991, 2021))
OVERLAP_YEARS = tuple(range(2013, 2021))
EE_PROJECT = "sf0c-ne-hls-20260729-5bef"
ERA5_LAND_MONTHLY = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA5_LAND_HOURLY = "ECMWF/ERA5_LAND/HOURLY"
EPSG6933_WKT = CRS.from_epsg(6933).to_wkt(version="WKT1_GDAL")
RAW_COLUMNS = (
    "jja_temperature_c",
    "jja_shortwave_mj_m2",
    "jja_rootzone_soil_moisture_m3_m3",
    "jja_vpd_p95_kpa",
)
COASTAL_NO_NATIVE_CELL_GRIDS = {
    "E6933_50_C235_R91",
    "E6933_50_C236_R92",
    "E6933_50_C237_R92",
}
ANOMALY_NAMES = {
    "jja_shortwave_mj_m2": "shortwave_z",
    "jja_temperature_c": "temperature_z",
    "jja_vpd_p95_kpa": "vpd_z",
    "jja_rootzone_soil_moisture_m3_m3": "rootzone_soil_moisture_z",
}


def formal_grids() -> list[str]:
    sample = pd.read_parquet(FORMAL_SAMPLE, columns=["grid_id"])
    grids = sorted(sample["grid_id"].astype(str).unique())
    if len(grids) != 514:
        raise RuntimeError(f"formal grid identity changed: {len(grids)}")
    return grids


def grid_collection(grids: list[str]) -> ee.FeatureCollection:
    features = []
    for grid_id in grids:
        parts = grid_id.split("_")
        column = int(parts[2][1:])
        row = int(parts[3][1:])
        geometry = ee.Geometry.Rectangle(
            [
                column * 50_000,
                row * 50_000,
                (column + 1) * 50_000,
                (row + 1) * 50_000,
            ],
            proj=EPSG6933_WKT,
            geodesic=False,
        )
        features.append(ee.Feature(geometry, {"grid_id": grid_id}))
    return ee.FeatureCollection(features)


def hourly_vpd(image: ee.Image) -> ee.Image:
    temperature = image.select("temperature_2m").subtract(273.15)
    dewpoint = image.select("dewpoint_temperature_2m").subtract(273.15)
    saturation = (
        temperature.multiply(17.27)
        .divide(temperature.add(237.3))
        .exp()
        .multiply(0.6108)
    )
    actual = (
        dewpoint.multiply(17.27)
        .divide(dewpoint.add(237.3))
        .exp()
        .multiply(0.6108)
    )
    return saturation.subtract(actual).max(0).rename("vpd_kpa")


def annual_climate_image(year: int) -> ee.Image:
    monthly = ee.ImageCollection(ERA5_LAND_MONTHLY).filterDate(
        f"{year}-06-01", f"{year}-09-01"
    )
    temperature = (
        monthly.select("temperature_2m")
        .mean()
        .subtract(273.15)
        .rename("jja_temperature_c")
    )
    shortwave = (
        monthly.select("surface_solar_radiation_downwards_sum")
        .sum()
        .divide(1_000_000)
        .rename("jja_shortwave_mj_m2")
    )
    soil_1 = monthly.select("volumetric_soil_water_layer_1").mean()
    soil_2 = monthly.select("volumetric_soil_water_layer_2").mean()
    soil_3 = monthly.select("volumetric_soil_water_layer_3").mean()
    rootzone = (
        soil_1.multiply(0.07)
        .add(soil_2.multiply(0.21))
        .add(soil_3.multiply(0.72))
        .rename("jja_rootzone_soil_moisture_m3_m3")
    )
    vpd = (
        ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate(f"{year}-06-01", f"{year}-09-01")
        .select(["temperature_2m", "dewpoint_temperature_2m"])
        .map(hourly_vpd)
        .reduce(ee.Reducer.percentile([95]))
        .rename("jja_vpd_p95_kpa")
    )
    return ee.Image.cat([temperature, shortwave, rootzone, vpd])


def recover_year(
    year: int, grids: list[str], collection: ee.FeatureCollection
) -> pd.DataFrame:
    image = annual_climate_image(year)
    if image.bandNames().getInfo() != list(RAW_COLUMNS):
        raise RuntimeError(f"ERA5-Land band identity changed for {year}")
    reduced = image.reduceRegions(
        collection=collection,
        reducer=ee.Reducer.mean(),
        scale=11_132,
        crs=EPSG6933_WKT,
        tileScale=2,
    )
    frame = pd.DataFrame(
        [feature["properties"] for feature in reduced.getInfo()["features"]]
    )[["grid_id", *RAW_COLUMNS]]
    frame["grid_id"] = frame["grid_id"].astype(str)
    missing = set(
        frame.loc[frame[list(RAW_COLUMNS)].isna().any(axis=1), "grid_id"]
    )
    if missing != COASTAL_NO_NATIVE_CELL_GRIDS:
        raise RuntimeError(
            f"unexpected native ERA5-Land missing grids for {year}: {sorted(missing)}"
        )
    filled_image = image.unmask(
        image.focalMean(radius=50_000, kernelType="circle", units="meters")
    )
    filled_collection = filled_image.reduceRegions(
        collection=grid_collection(sorted(COASTAL_NO_NATIVE_CELL_GRIDS)),
        reducer=ee.Reducer.mean(),
        scale=11_132,
        crs=EPSG6933_WKT,
        tileScale=2,
    )
    filled = pd.DataFrame(
        [feature["properties"] for feature in filled_collection.getInfo()["features"]]
    ).set_index("grid_id")
    frame = frame.set_index("grid_id")
    frame.loc[sorted(COASTAL_NO_NATIVE_CELL_GRIDS), list(RAW_COLUMNS)] = filled.loc[
        sorted(COASTAL_NO_NATIVE_CELL_GRIDS), list(RAW_COLUMNS)
    ]
    frame = frame.reset_index()
    frame["year"] = year
    frame = frame[["grid_id", "year", *RAW_COLUMNS]].sort_values("grid_id")
    if len(frame) != len(grids) or frame[list(RAW_COLUMNS)].isna().any().any():
        raise RuntimeError(f"incomplete ERA5-Land support for {year}")
    if frame["grid_id"].tolist() != grids:
        raise RuntimeError(f"formal grid order changed for {year}")
    return frame.reset_index(drop=True)


def validation_against_current(reference: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_parquet(CURRENT_CLIMATE)[
        ["grid_id", "year", *RAW_COLUMNS]
    ].copy()
    current["grid_id"] = current["grid_id"].astype(str)
    overlap = reference.loc[reference["year"].isin(OVERLAP_YEARS)].merge(
        current.loc[current["year"].isin(OVERLAP_YEARS)],
        on=["grid_id", "year"],
        suffixes=("_recovered", "_current"),
        validate="one_to_one",
    )
    if len(overlap) != 514 * len(OVERLAP_YEARS):
        raise RuntimeError("2013--2020 overlap support is incomplete")
    rows = []
    for variable in RAW_COLUMNS:
        difference = (
            overlap[f"{variable}_recovered"] - overlap[f"{variable}_current"]
        ).abs()
        rows.append(
            {
                "variable": variable,
                "overlap_year_start": min(OVERLAP_YEARS),
                "overlap_year_end": max(OVERLAP_YEARS),
                "comparison_n": int(len(difference)),
                "maximum_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
            }
        )
    return pd.DataFrame(rows)


def climate_normals(reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in reference.groupby("grid_id", sort=True):
        row: dict[str, str | float | int] = {
            "grid_id": grid_id,
            "reference_year_start": YEARS[0],
            "reference_year_end": YEARS[-1],
            "reference_year_n": len(YEARS),
        }
        for variable in RAW_COLUMNS:
            row[f"{variable}_mean_1991_2020"] = float(group[variable].mean())
            row[f"{variable}_sd_1991_2020"] = float(group[variable].std(ddof=1))
        rows.append(row)
    output = pd.DataFrame(rows)
    standard_deviation_columns = [
        column for column in output.columns if column.endswith("_sd_1991_2020")
    ]
    if output[standard_deviation_columns].le(0.0).any().any():
        raise RuntimeError("one or more 1991--2020 climate standard deviations are invalid")
    return output


def restandardize_current(normals: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_parquet(CURRENT_CLIMATE)[
        ["grid_id", "year", *RAW_COLUMNS]
    ].copy()
    current["grid_id"] = current["grid_id"].astype(str)
    output = current.merge(normals, on="grid_id", validate="many_to_one")
    for raw_variable, anomaly in ANOMALY_NAMES.items():
        output[anomaly] = (
            output[raw_variable] - output[f"{raw_variable}_mean_1991_2020"]
        ) / output[f"{raw_variable}_sd_1991_2020"]
    columns = ["grid_id", *RAW_COLUMNS[:3], "year", RAW_COLUMNS[3], *ANOMALY_NAMES.values()]
    return output[columns].sort_values(["grid_id", "year"]).reset_index(drop=True)


def main() -> None:
    if (OUTPUT / "analysis_metadata.json").exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    grids = formal_grids()
    ee.Initialize(project=EE_PROJECT)
    collection = grid_collection(grids)

    frames = []
    for year in YEARS:
        checkpoint = CHECKPOINTS / f"era5_land_jja_{year}.parquet"
        if checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
            print(f"{year}: checkpoint", flush=True)
        else:
            print(f"{year}: recovering ERA5-Land", flush=True)
            frame = recover_year(year, grids, collection)
            temporary = checkpoint.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.rename(checkpoint)
        frames.append(frame)

    reference = pd.concat(frames, ignore_index=True).sort_values(
        ["grid_id", "year"]
    )
    if len(reference) != 514 * len(YEARS):
        raise RuntimeError("1991--2020 reference table is incomplete")
    validation = validation_against_current(reference)
    normals = climate_normals(reference)
    restandardized = restandardize_current(normals)

    temporary_root = Path(tempfile.mkdtemp(prefix="climate_normals_1991_2020_"))
    try:
        reference.to_parquet(
            temporary_root / "grid_year_climate_1991_2020.parquet",
            index=False,
            compression="zstd",
        )
        normals.to_parquet(
            temporary_root / "grid_climate_normals_1991_2020.parquet",
            index=False,
            compression="zstd",
        )
        restandardized.to_parquet(
            temporary_root / "grid_year_climate_with_1991_2020_anomalies.parquet",
            index=False,
            compression="zstd",
        )
        validation.to_csv(
            temporary_root / "overlap_reproduction_2013_2020.csv", index=False
        )
        metadata = {
            "status": "COMPLETE_CLIMATE_NORMALS_1991_2020",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "grid_n": 514,
            "reference_year_start": YEARS[0],
            "reference_year_end": YEARS[-1],
            "reference_year_n": len(YEARS),
            "response_year_start": 2013,
            "response_year_end": 2025,
            "raw_variables": list(RAW_COLUMNS),
            "sources": {
                "temperature_shortwave_rootzone": ERA5_LAND_MONTHLY,
                "vpd_p95": ERA5_LAND_HOURLY,
            },
            "jja_definitions": {
                "temperature": "mean of June--August monthly mean 2-m temperature",
                "shortwave": "sum of June--August monthly downward shortwave totals",
                "rootzone_soil_moisture": "June--August mean of ERA5-Land layers 1--3 weighted 0.07, 0.21, and 0.72",
                "vpd": "95th percentile of June--August hourly VPD derived from 2-m air and dewpoint temperature",
            },
            "coastal_fill": "masked cells filled from a 50-km focal mean, matching the recovered 2013--2025 definition",
            "coastal_fill_grid_ids": sorted(COASTAL_NO_NATIVE_CELL_GRIDS),
            "earth_engine_export_tasks_started": False,
            "overlap_validation": validation.to_dict("records"),
            "formal_reports_updated": False,
        }
        (temporary_root / "analysis_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        for source in temporary_root.iterdir():
            target = OUTPUT / source.name
            if target.exists():
                raise RuntimeError(f"refusing to overwrite reference output: {target}")
            source.rename(target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    print(f"COMPLETE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
