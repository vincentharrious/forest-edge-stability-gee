"""Offline checks: run with python check_engineering.py; no Earth Engine calls."""
import datetime as dt
import importlib.util
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("extract", Path(__file__).parent / "earth_engine/extract_observations.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)

# Exhaustive 16-bit DN equivalence; no ecological assumptions or data.
for dn in range(65536):
    assert (7273 <= dn <= 43636) == (0 <= dn * .0000275 - .2 <= 1)

points = [dict(pixel_id="example", longitude=127., latitude=45.)]
transform = [30, 0, 500000, 0, -30, 5000000]
key = extract.checkpoint_key(points, 2020, "EPSG:32652", transform)
assert key != extract.checkpoint_key(points, 2021, "EPSG:32652", transform)
assert key != extract.checkpoint_key(points, 2020, "EPSG:32651", transform)
millis = dt.datetime(2020, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1000
row = [10000, 20000, millis, 8, 110030, 0, 0, 0]
response = {"features": [{"properties": {"pixel_id": "example", "first": [row]}}]}
assert extract.validate_response(response, points, 2020) == 1
try:
    extract.validate_response({**response, "nextPageToken": "more"}, points, 2020)
except ValueError:
    pass
else:
    raise AssertionError("Pagination must not silently truncate data")

with tempfile.TemporaryDirectory() as directory:
    output = Path(directory)
    destination = output / f"2020-{key}.json"
    extract.write_json(destination, dict(status="complete", input_digest=key, response=response))
    # No EE client is supplied: resuming valid saved data must not contact EE,
    # even if no profile was ever persisted.
    extract.run_one(None, points, 2020, "EPSG:32652", transform, output)
print("Passed: DN equivalence, input identity, response validation and checkpoint reuse")
