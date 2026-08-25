# Forest-Edge Canopy Stability under Climate Extremes

This repository contains the Google Earth Engine processing code and downstream analysis scripts for a study investigating how forest fragmentation and edge exposure affect the year-to-year stability of forest canopy functioning under heat, drought, and compound climate extremes.

## Project overview

Forest fragmentation exposes over 70% of the world's remaining forests to edge effects within 1 km of a non-forest boundary. This project uses 30 m Harmonized Landsat and Sentinel-2 (HLS) observations and ERA5-Land climate records to quantify where fragmented and edge-exposed forests are most vulnerable to climate extremes, and how that vulnerability has changed over the past decade (2013–2025).

### Current status

- **Regional analysis (Northeast China)**: Completed and validated. 345,033 forest pixels across 514 sampling grids; 4,156,725 annual observations.
- **Global climate preprocessing**: Completed. 394,488 hourly ERA5-Land timestamps across 675,939 grid cells; approximately 50.8 million cell-level heatwave events identified (1981–2025).
- **Global canopy extraction**: Pending. Up to 8,593 candidate 100 × 100 km grid cells across six major forest biomes.

### Key findings (regional pilot)

- Canopy temporal stability 30 m from forest edges was 20.9% lower than at 480 m.
- Year-to-year variability was 23.3% greater near edges.
- Forests with denser canopy cover were less vulnerable; edges adjacent to wider open areas showed greater instability.

## Repository structure

```
earth_engine/
├── canopy_extraction_pilot.py     # 30 m annual NIRv extraction — 4-patch canary (validated)
├── canopy_extraction_global.py    # 30 m annual NIRv extraction — full 48-patch scope
└── climate_preprocessing.py       # ERA5-Land 1991–2020 JJA climate normals recovery

analysis/
├── fit_nonlinear_main_effect_check.py         # Nonlinear climate main-effect models
├── project_observed_climate_states.py         # Climate-state projection onto distance curves
├── fit_five_year_full_curve_controls.py       # Five-year window stability curve analysis
└── fit_regional_stability_time_trend.py       # Regional stability time-trend analysis
```

## Earth Engine processing pipeline

### Stage 1: Climate preprocessing (completed)

`earth_engine/climate_preprocessing.py` recovers ERA5-Land JJA (June–August) climate normals for the 1991–2020 reference period across the 514 formal sampling grids. Variables include:
- Mean 2-m air temperature
- Cumulative downward shortwave radiation
- Root-zone soil moisture (layers 1–3, depth-weighted)
- 95th-percentile vapor pressure deficit (VPD), derived from hourly 2-m air and dewpoint temperature

The script uses checkpoint-based restart to avoid reprocessing completed years.

### Stage 2: 30 m canopy extraction (regional pilot completed; global pending)

`earth_engine/canopy_extraction_pilot.py` and `canopy_extraction_global.py` extract annual growing-season median NIRv (near-infrared reflectance of vegetation) at 30 m resolution from the HLS L30 collection.

Key processing steps:
- Cloud, cloud-shadow, snow, and high-aerosol masking via the HLS Fmask band
- Annual scene count, distance-to-edge, and median NIRv computed per pixel per year
- Pixels filtered by minimum valid-year count (≥10 of 13 years) and distance range (30–480 m from forest edge)
- Each forest patch processed as an independently resumable spatial unit
- Results exported as compact CSV summaries to Cloud Storage

### Optimizations implemented

- Spatial partitioning into fixed, independently resumable units with checkpoint-based restart
- `filterBounds`, date, and band pre-filtering before image collection construction
- Single shared cloud/shadow/snow validity mask reused across all analyses
- Compatible reducers combined to avoid redundant collection evaluation
- Memory-intensive annual calculations split into seasonal sub-calculations (verified to reproduce annual results exactly)
- Compact annual summary export instead of hourly arrays
- Controlled batch submission concurrency
- Preflight and cloud-state validation before any submission to prevent duplicate or overwriting tasks

A matched optimization test showed the optimized implementation reduced Earth Engine computation by 51.9% while reproducing the same scientific output.

## Compute requirements

| Stage | Scope | EECU-hours |
|---|---|---|
| Climate preprocessing | 675,939 grid cells × 45 years | Completed |
| Canopy extraction (NE China) | 163 processing units | ~289 |
| Canopy extraction (global, estimated) | ~8,593 processing units | ~15,000 |

The estimated global canopy extraction requirement is approximately 15× the monthly Contributor Tier quota (1,000 EECU-hours).

## Data sources

| Dataset | Earth Engine ID | Resolution | Period |
|---|---|---|---|
| Harmonized Landsat Sentinel-2 (L30) | `NASA/HLS/HLSL30/v002` | 30 m | 2013–2025 |
| ERA5-Land Monthly | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | ~11 km | 1981–2025 |
| ERA5-Land Hourly | `ECMWF/ERA5_LAND/HOURLY` | ~11 km | 1981–2025 |
| Global Forest Change | Hansen et al. 2013 (via derived lattice) | 30 m | 2000–2023 |

## Related publication

Zhou, Z.; et al. (2023). *Remote Sensing*, 15(5), 1335. DOI: [10.3390/rs15051335](https://doi.org/10.3390/rs15051335)

## Affiliation

School of Ecology, Northeast Forestry University, Harbin, China.

## License

This code is released for noncommercial research use.
