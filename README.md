# Forest temporal stability: Earth Engine processing example

Noncommercial research at the School of Ecology, Northeast Forestry University.
Our planned study assesses forest functional stability over 2001–2024 across
global forest environments. Northeast China is the current regional development
and processing test case. Global production has not been completed.

This repository provides an **engineering example**, adapted from our regional
Landsat observation-extraction workflow, to make its computational design
reviewable. It is not a complete scientific reproduction package. The current
release contains no unpublished effect estimates, statistical models, research
sample coordinates, credentials, private assets, or manuscript results. No
publication is claimed for this project.

## Contents

- `earth_engine/extract_observations.py`: one bounded point-batch/year extraction,
  input-aware checkpoints, QA checks and raw profiler capture.
- `check_engineering.py`: offline checks that consume no Earth Engine resources.
- `engineering_evidence.json`: aggregate engineering counts from a completed
  regional run, with explicit limits on the compute accounting.

## Why Earth Engine is needed

The processing stage accesses Landsat Collection 2 Level 2 imagery from Landsat
5, 7, 8 and 9 at 30 m for 24 years. It filters observations, applies consistent
quality screening, and extracts red/NIR observations and provenance at supplied
sample locations. Extracted observations can be reused locally for temporal
aggregation and statistical analysis without repeatedly scanning satellite
collections in Earth Engine. Climate and forest-context inputs are separate
workloads and are not implemented in this example.

The intended research output is a documented, geographically broad sampled
dataset and comparative assessment of forest stability, supporting research on
forest conservation and climate adaptation. This does not promise a wall-to-wall
global 30 m map or claim that management agencies already use the results.

## Implemented computational choices

1. Filter dates, location and required bands before constructing the extraction.
2. Reuse one QA mask for red/NIR and provenance bands; retain QA metadata.
3. Apply exact integer-DN reflectance bounds before transfer. Convert retained
   red/NIR values locally with `reflectance = DN * 0.0000275 - 0.2`.
4. Transfer selected observations at sampled locations, rather than full imagery.
   Per-observation values are retained so alternative temporal summaries do not
   require another cloud extraction.
5. Keep each request to one compact spatial unit and one year. This public
   example submits one request at a time and never launches a global job set.
6. Key checkpoints by input coordinates, year, projection and code contract;
   write completed data atomically before profiler finalization. Missing
   profiling does not trigger re-extraction of saved observations.
7. Reject unexpected pagination, missing samples and invalid observations.
   Errors propagate for review; this example does not automatically resubmit.

The production workflow also used bounded concurrency and smaller point batches
when memory limits required splitting. Those orchestration features are not
included in this minimal example. No percentage compute saving is claimed for
this release.

## Recorded regional workload

The completed 2001–2024 extraction covered 39,762 requested sample locations in
151 processing grids: 3,624 grid-year files and 21,310,299 retained observation
rows. Points need not have a valid observation in every year. These are processing
counts, not the final scientific analysis sample.

Available production/check profiles sum to **248.57 EECU-hours**, but **99
successful requests lacked profiles**. Therefore this is incomplete compute
accounting, not the total cost. A small regional pilot predicted approximately
365.86 EECU-hours for that regional workload; the forecast and partial measured
total are different quantities. A project-wide monthly snapshot on September 17,
2026 recorded 879.07 EECU-hours and includes other work. It is not a current
balance or evidence that the project had exhausted its monthly quota.

Global cost cannot be inferred by area alone: cloud cover, scene density,
sampling density and forest context vary. Expansion will be staged, with
representative benchmarks and monthly monitoring before scaling. The regional
workflow already consumes a substantial share of the Contributor allowance while
global coverage, additional context and validation remain ahead.

## Run a small example

Python 3.10+ is recommended. Authenticate with your own registered noncommercial
Earth Engine project; no project or private assets are embedded here.

```sh
python -m pip install -r requirements.txt
earthengine authenticate
python check_engineering.py
```

Supply your own CSV with columns `pixel_id,longitude,latitude`, containing
1–500 unique IDs from a compact region. Coordinates are WGS84. Specify a metric
projected CRS and its 30 m grid transform to match your sample grid. The following
is only an example projection, not the research grid:

```sh
python earth_engine/extract_observations.py \
  --project YOUR_PROJECT_ID --points YOUR_POINTS.csv --year 2020 \
  --crs EPSG:32652 --transform 30 0 500000 0 -30 5000000
```

This validates inputs and prints the plan without contacting Earth Engine. Add
`--execute` to perform the metered extraction. A full-year array can still exceed
memory in dense archives; reduce the point batch when necessary. A single point
or small set is the appropriate first live test. Run only one process per output
directory; this example is sequential and has no concurrent-writer lock.

Each feature's `first` array contains rows ordered as:
`red_dn, nir_dn, time_ms, sensor, pathrow, QA_PIXEL, QA_RADSAT, sensor_atmosphere`.
Masked points remain in the response but may have no observation array.
Atmospheric metadata differ between sensor families and require sensor-specific
interpretation. `-9999` denotes missing atmospheric metadata.

Raw `.profile.txt` output is retained for inspection. Cross-check accounting with
Cloud Monitoring; wall-clock time is not EECU usage, and a missing profile is not
zero consumption. If profiling fails after data persistence, the error remains
visible and the next invocation reuses the completed checkpoint.

## Validation and scope

The source regional workflow completed a pilot comparison against a floating-
reflectance implementation before production. For this public adaptation, local
checks verify integer-bound equivalence across all 16-bit DN values, checkpoint
identity, response completeness and reuse of saved data without a profiler.
The public adaptation has not been run as a new live EE extraction; no extra
research compute was consumed to publish it. Do not interpret offline checks as
end-to-end cloud validation or multi-sensor scientific harmonization.

## License

As in the previous repository, this code is provided for noncommercial research
use. Source imagery remains subject to its respective terms.
