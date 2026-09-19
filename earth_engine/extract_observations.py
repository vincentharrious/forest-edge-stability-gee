"""Small, sequential Landsat extraction example adapted from our regional workflow.

No forest classification, edge-effect model, or unpublished research outputs.
Requires an explicitly supplied EE project, sample CSV, and projection.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SENSORS = {
    5: ("LANDSAT/LT05/C02/T1_L2", "SR_B3", "SR_B4"),
    7: ("LANDSAT/LE07/C02/T1_L2", "SR_B3", "SR_B4"),
    8: ("LANDSAT/LC08/C02/T1_L2", "SR_B4", "SR_B5"),
    9: ("LANDSAT/LC09/C02/T1_L2", "SR_B4", "SR_B5"),
}
VERSION = "landsat-observation-example-v1"


def load_points(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) > 500:
        raise ValueError("Supply 1–500 points from one compact processing unit")
    points = []
    for row in rows:
        lon, lat = float(row["longitude"]), float(row["latitude"])
        if not (-180 <= lon <= 180 and -90 <= lat <= 90) or not row["pixel_id"]:
            raise ValueError("Invalid coordinates or empty pixel_id")
        points.append(dict(pixel_id=row["pixel_id"], longitude=lon, latitude=lat))
    if len({p["pixel_id"] for p in points}) != len(points):
        raise ValueError("pixel_id must be unique")
    return sorted(points, key=lambda p: p["pixel_id"])


def checkpoint_key(points, year, crs, transform):
    contract = dict(version=VERSION, points=points, year=year, crs=crs, transform=transform)
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def build_expression(ee, points, year, crs, transform):
    features = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p["longitude"], p["latitude"]]),
                   {"pixel_id": p["pixel_id"]}) for p in points
    ])
    bounds = features.geometry().bounds(30)
    collection = ee.ImageCollection([])
    for sensor, (asset, red, nir) in SENSORS.items():
        atmosphere = "SR_QA_AEROSOL" if sensor in (8, 9) else "SR_ATMOS_OPACITY"
        source = (ee.ImageCollection(asset)
                  .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                  .filterBounds(bounds)
                  .select([red, nir, "QA_PIXEL", "QA_RADSAT", atmosphere]))

        def prepare(image, sensor=sensor, red=red, nir=nir, atmosphere=atmosphere):
            raw = image.select([red, nir], ["red_dn", "nir_dn"])
            qa, sat = image.select("QA_PIXEL"), image.select("QA_RADSAT")
            # Equivalent integer bounds for 0 <= DN * 0.0000275 - 0.2 <= 1.
            valid = (qa.bitwiseAnd(63).eq(0).And(sat.eq(0))
                     .And(raw.gte(7273).And(raw.lte(43636)).reduce(ee.Reducer.min())))
            auxiliary = ee.Image.constant([
                ee.Number(image.get("system:time_start")), sensor,
                ee.Number(image.get("WRS_PATH")).multiply(1000).add(
                    ee.Number(image.get("WRS_ROW"))),
            ]).rename(["time_ms", "sensor", "pathrow"]).toInt64()
            return (raw.addBands(auxiliary).addBands(qa).addBands(sat)
                    .addBands(image.select(atmosphere).rename("sensor_atmosphere").unmask(-9999))
                    .updateMask(valid))

        collection = collection.merge(source.map(prepare))
    return collection.toArray().reduceRegions(
        collection=features, reducer=ee.Reducer.first(), crs=crs,
        crsTransform=transform, tileScale=4,
    )


def validate_response(response, points, year):
    import datetime as dt
    if response.get("nextPageToken"):
        raise ValueError("Unexpected pagination: reduce the point batch")
    features = response["features"]
    expected = sorted(p["pixel_id"] for p in points)
    if sorted(f["properties"]["pixel_id"] for f in features) != expected:
        raise ValueError("Response does not match the requested sample")
    seen, count = set(), 0
    for feature in features:
        prop = feature["properties"]
        for row in prop.get("first") or []:
            if len(row) != 8:
                raise ValueError("Unexpected observation schema")
            red, nir, millis, sensor, pathrow, qa, sat, atmosphere = row
            if not (7273 <= red <= 43636 and 7273 <= nir <= 43636):
                raise ValueError("Reflectance bounds check failed")
            if int(qa) & 63 or sat != 0 or sensor not in SENSORS:
                raise ValueError("QA or sensor check failed")
            if dt.datetime.fromtimestamp(millis / 1000, dt.timezone.utc).year != year:
                raise ValueError("Observation outside requested year")
            key = (prop["pixel_id"], millis, sensor, pathrow)
            if key in seen:
                raise ValueError("Duplicate observation")
            seen.add(key)
            count += 1
    return count


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def run_one(ee, points, year, crs, transform, output):
    digest = checkpoint_key(points, year, crs, transform)
    destination = output / f"{year}-{digest}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["input_digest"] != digest or saved["status"] != "complete":
            raise ValueError("Invalid checkpoint")
        validate_response(saved["response"], points, year)
        print(f"{year}: reused completed extraction")
        return
    profile = io.StringIO()
    try:
        with ee.profilePrinting(destination=profile):
            response = ee.data.computeFeatures({
                "expression": build_expression(ee, points, year, crs, transform),
                "pageSize": 1000, "workloadTag": "forest-stability-engineering",
            })
            count = validate_response(response, points, year)
            # Save successful data BEFORE profiling exits. A missing profile must
            # not cause another extraction of successfully saved observations.
            write_json(destination, dict(status="complete", input_digest=digest,
                                         observations=count, response=response))
    finally:
        destination.with_suffix(".profile.txt").write_text(profile.getvalue())
    print(f"{year}: saved {count} observations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--crs", required=True, help="Projected CRS matching the supplied sample grid")
    parser.add_argument("--transform", required=True, nargs=6, type=float,
                        metavar=("SX", "HX", "TX", "HY", "SY", "TY"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--execute", action="store_true", help="Make the metered EE request")
    args = parser.parse_args()
    points = load_points(args.points)
    if not 2001 <= args.year <= 2024:
        parser.error("This example covers 2001–2024")
    if not all(math.isfinite(x) for x in args.transform):
        parser.error("Transform must contain finite values")
    sx, hx, _, hy, sy, _ = args.transform
    if sx * sy - hx * hy == 0:
        parser.error("Transform must be invertible")
    print(json.dumps(dict(points=len(points), year=args.year, project=args.project,
                          execute=args.execute)))
    if not args.execute:
        return
    import ee
    ee.Initialize(project=args.project)
    ee.data.setDeadline(180000)
    ee.data.setMaxRetries(0)
    args.output.mkdir(parents=True, exist_ok=True)
    run_one(ee, points, args.year, args.crs, args.transform, args.output)


if __name__ == "__main__":
    main()
