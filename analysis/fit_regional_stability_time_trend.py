#!/usr/bin/env python3
"""Estimate change in regional rolling temporal stability over 2013--2025."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "analysis_inputs" / "annual_climate_state" / "formal_sample.parquet"
OUTPUT = ROOT / "outputs" / "core_results" / "08_regional_stability_trend"

YEARS = np.arange(2013, 2026, dtype=np.int32)
WINDOW_N = 7
WINDOW_STARTS = np.arange(0, len(YEARS) - WINDOW_N + 1, dtype=np.int32)
WINDOW_CENTERS = YEARS[WINDOW_STARTS + WINDOW_N // 2]
TIME = WINDOW_CENTERS.astype(np.float64) - float(WINDOW_CENTERS.mean())
FREE_WINDOW_N = len(WINDOW_CENTERS) - 1
DISTANCE_NODES_M = np.asarray([30.0, 60.0, 120.0, 240.0, 480.0])
DENSE_DISTANCE_M = np.geomspace(30.0, 480.0, 97)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260823


def configure(window_years: int, output: Path) -> None:
    global OUTPUT, WINDOW_N, WINDOW_STARTS, WINDOW_CENTERS, TIME, FREE_WINDOW_N
    OUTPUT = output
    WINDOW_N = window_years
    WINDOW_STARTS = np.arange(0, len(YEARS) - WINDOW_N + 1, dtype=np.int32)
    WINDOW_CENTERS = YEARS[WINDOW_STARTS + WINDOW_N // 2]
    TIME = WINDOW_CENTERS.astype(np.float64) - float(WINDOW_CENTERS.mean())
    FREE_WINDOW_N = len(WINDOW_CENTERS) - 1


def node_basis(value: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    interval = np.searchsorted(nodes, value, side="right") - 1
    interval = np.clip(interval, 0, len(nodes) - 2)
    fraction = (value - nodes[interval]) / (nodes[interval + 1] - nodes[interval])
    result = np.zeros((len(value), len(nodes)), dtype=np.float64)
    row = np.arange(len(value))
    result[row, interval] = 1.0 - fraction
    result[row, interval + 1] = fraction
    return result


def group_sum(value: np.ndarray, group: np.ndarray, group_n: int) -> np.ndarray:
    return np.bincount(group, weights=value, minlength=group_n).astype(np.float64)


def cross_products_by_grid(
    design: np.ndarray,
    outcomes: np.ndarray,
    weight: np.ndarray,
    grid: np.ndarray,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    parameter_n = design.shape[1]
    outcome_n = outcomes.shape[1]
    xtwx = np.zeros((grid_n, parameter_n, parameter_n), dtype=np.float64)
    xtwy = np.zeros((grid_n, parameter_n, outcome_n), dtype=np.float64)
    for left in range(parameter_n):
        for right in range(left, parameter_n):
            value = group_sum(
                weight * design[:, left] * design[:, right], grid, grid_n
            )
            xtwx[:, left, right] = value
            xtwx[:, right, left] = value
        for outcome in range(outcome_n):
            xtwy[:, left, outcome] = group_sum(
                weight * design[:, left] * outcomes[:, outcome], grid, grid_n
            )
    return xtwx, xtwy


def fit_from_cross_products(
    xtwx: np.ndarray, xtwy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cross = xtwx.sum(axis=0)
    inverse = np.linalg.inv(cross)
    beta = inverse @ xtwy.sum(axis=0)
    return beta, inverse


def joint_cluster_covariance(
    design: np.ndarray,
    outcomes: np.ndarray,
    weight: np.ndarray,
    grid: np.ndarray,
    grid_n: int,
    beta: np.ndarray,
    inverse: np.ndarray,
) -> np.ndarray:
    parameter_n, outcome_n = beta.shape
    scores = np.zeros((grid_n, parameter_n * outcome_n), dtype=np.float64)
    residual = outcomes - design @ beta
    for outcome in range(outcome_n):
        start = outcome * parameter_n
        for parameter in range(parameter_n):
            scores[:, start + parameter] = group_sum(
                weight
                * design[:, parameter]
                * residual[:, outcome],
                grid,
                grid_n,
            )
    bread = np.zeros(
        (parameter_n * outcome_n, parameter_n * outcome_n), dtype=np.float64
    )
    for outcome in range(outcome_n):
        start = outcome * parameter_n
        bread[start : start + parameter_n, start : start + parameter_n] = inverse
    correction = grid_n / (grid_n - 1.0)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return 0.5 * (covariance + covariance.T)


def p_from_bootstrap(draw: np.ndarray) -> float:
    lower = (np.sum(draw <= 0.0) + 1.0) / (len(draw) + 1.0)
    upper = (np.sum(draw >= 0.0) + 1.0) / (len(draw) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def window_effects(
    beta: np.ndarray,
) -> np.ndarray:
    node_n = len(DISTANCE_NODES_M)
    free = beta[: FREE_WINDOW_N * node_n].reshape(FREE_WINDOW_N, node_n, 2)
    return np.concatenate((free, -free.sum(axis=0, keepdims=True)), axis=0)


def stability_log_difference(
    mean_first: np.ndarray,
    variance_first: np.ndarray,
    mean_last: np.ndarray,
    variance_last: np.ndarray,
) -> np.ndarray:
    if (
        np.any(mean_first <= 0.0)
        or np.any(mean_last <= 0.0)
        or np.any(variance_first <= 0.0)
        or np.any(variance_last <= 0.0)
    ):
        raise RuntimeError("time-trend projection left the positive support")
    return (
        np.log(mean_last)
        - 0.5 * np.log(variance_last)
        - np.log(mean_first)
        + 0.5 * np.log(variance_first)
    )


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUTPUT}")

    columns = ["pixel_id", "grid_id", "base_weight", "d_between_m"]
    columns += [f"nirv_{year}" for year in YEARS]
    columns += [f"distance_m_{year}" for year in YEARS]
    columns += [f"valid_scene_n_{year}" for year in YEARS]
    sample = pd.read_parquet(INPUT, columns=columns)

    nirv = np.column_stack(
        [sample[f"nirv_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    annual_distance = np.column_stack(
        [sample[f"distance_m_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    scene = np.column_stack(
        [sample[f"valid_scene_n_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    formal = sample["d_between_m"].between(30.0, 480.0).to_numpy()
    valid = (
        np.isfinite(nirv)
        & np.isfinite(annual_distance)
        & (annual_distance > 0.0)
        & (scene >= 1.0)
    )
    balanced = formal & valid.all(axis=1)
    if int(balanced.sum()) != 129_307:
        raise RuntimeError(f"balanced support changed: {int(balanced.sum())}")

    nirv = nirv[balanced]
    scene = scene[balanced]
    distance = sample.loc[balanced, "d_between_m"].to_numpy(np.float64)
    pixel_weight = sample.loc[balanced, "base_weight"].to_numpy(np.float64)
    grid_name = sample.loc[balanced, "grid_id"].astype(str).to_numpy()
    grid, grid_levels = pd.factorize(grid_name, sort=True)
    grid = grid.astype(np.int32)
    grid_n = len(grid_levels)
    basis = node_basis(np.log10(distance), np.log10(DISTANCE_NODES_M))

    window_mean = np.empty((len(distance), len(WINDOW_STARTS)), dtype=np.float64)
    window_variance = np.empty_like(window_mean)
    scene_one = np.empty_like(window_mean)
    scene_two_three = np.empty_like(window_mean)
    for index, start in enumerate(WINDOW_STARTS):
        stop = start + WINDOW_N
        current = nirv[:, start:stop]
        current_scene = scene[:, start:stop]
        window_mean[:, index] = current.mean(axis=1)
        window_variance[:, index] = current.var(axis=1, ddof=1)
        scene_one[:, index] = (current_scene == 1.0).mean(axis=1)
        scene_two_three[:, index] = (
            (current_scene >= 2.0) & (current_scene <= 3.0)
        ).mean(axis=1)
    if np.any(window_variance <= 0.0):
        raise RuntimeError("one or more rolling windows have zero temporal variance")

    window_code = np.zeros((len(WINDOW_CENTERS), FREE_WINDOW_N), dtype=np.float64)
    window_code[:FREE_WINDOW_N] = np.eye(FREE_WINDOW_N)
    window_code[-1] = -1.0
    primary = (
        basis[:, None, None, :]
        * window_code[None, :, :, None]
    ).reshape(len(distance), len(WINDOW_CENTERS), -1)
    controls = np.stack((scene_one, scene_two_three), axis=2)
    controls -= controls.mean(axis=1, keepdims=True)
    design = np.concatenate((primary, controls), axis=2).reshape(
        -1, FREE_WINDOW_N * len(DISTANCE_NODES_M) + 2
    )
    outcomes_raw = np.stack((window_mean, window_variance), axis=2)
    outcomes = outcomes_raw - outcomes_raw.mean(axis=1, keepdims=True)
    outcomes = outcomes.reshape(-1, 2)
    weight = np.repeat(pixel_weight, len(WINDOW_STARTS))
    weight /= weight.mean()
    grid_long = np.repeat(grid, len(WINDOW_STARTS))

    xtwx_grid, xtwy_grid = cross_products_by_grid(
        design, outcomes, weight, grid_long, grid_n
    )
    beta, inverse = fit_from_cross_products(xtwx_grid, xtwy_grid)
    joint_cluster_covariance(
        design, outcomes, weight, grid_long, grid_n, beta, inverse
    )

    center_outcomes = outcomes_raw.mean(axis=1)
    center_xtwx_grid, center_xtwy_grid = cross_products_by_grid(
        basis, center_outcomes, pixel_weight, grid, grid_n
    )
    center_beta, _ = fit_from_cross_products(center_xtwx_grid, center_xtwy_grid)

    dense_basis = node_basis(
        np.log10(DENSE_DISTANCE_M), np.log10(DISTANCE_NODES_M)
    )
    effects = window_effects(beta)
    dense_mean = np.stack(
        [dense_basis @ (center_beta[:, 0] + current[:, 0]) for current in effects]
    )
    dense_variance = np.stack(
        [dense_basis @ (center_beta[:, 1] + current[:, 1]) for current in effects]
    )
    log_change = stability_log_difference(
        dense_mean[0], dense_variance[0], dense_mean[-1], dense_variance[-1]
    )

    weight_grid = group_sum(pixel_weight, grid, grid_n)
    basis_weight_grid = np.column_stack(
        [
            group_sum(pixel_weight * basis[:, column], grid, grid_n)
            for column in range(basis.shape[1])
        ]
    )
    average_basis = basis_weight_grid.sum(axis=0) / weight_grid.sum()
    regional_mean = np.asarray(
        [average_basis @ (center_beta[:, 0] + current[:, 0]) for current in effects]
    )
    regional_variance = np.asarray(
        [average_basis @ (center_beta[:, 1] + current[:, 1]) for current in effects]
    )
    regional_log_stability = np.log(regional_mean) - 0.5 * np.log(
        regional_variance
    )
    regional_log_change = float(
        regional_log_stability[-1] - regional_log_stability[0]
    )
    regional_linear_slope = float(np.polyfit(TIME, regional_log_stability, 1)[0])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    regional_bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    regional_slope_bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    regional_window_bootstrap = np.empty(
        (BOOTSTRAP_DRAWS, len(WINDOW_CENTERS)), dtype=np.float64
    )
    node_bootstrap = np.empty(
        (BOOTSTRAP_DRAWS, len(DISTANCE_NODES_M)), dtype=np.float64
    )
    dense_bootstrap = np.empty(
        (BOOTSTRAP_DRAWS, len(DENSE_DISTANCE_M)), dtype=np.float64
    )
    node_window_bootstrap = np.empty(
        (
            BOOTSTRAP_DRAWS,
            len(WINDOW_CENTERS),
            len(DISTANCE_NODES_M),
        ),
        dtype=np.float64,
    )
    dense_window_bootstrap = np.empty(
        (
            BOOTSTRAP_DRAWS,
            len(WINDOW_CENTERS),
            len(DENSE_DISTANCE_M),
        ),
        dtype=np.float64,
    )
    for draw in range(BOOTSTRAP_DRAWS):
        count = rng.multinomial(
            grid_n, np.full(grid_n, 1.0 / grid_n, dtype=np.float64)
        ).astype(np.float64)
        current_xtwx = np.einsum("g,gij->ij", count, xtwx_grid, optimize=True)
        current_xtwy = np.einsum("g,gij->ij", count, xtwy_grid, optimize=True)
        current_beta = np.linalg.solve(current_xtwx, current_xtwy)
        current_center_xtwx = np.einsum(
            "g,gij->ij", count, center_xtwx_grid, optimize=True
        )
        current_center_xtwy = np.einsum(
            "g,gij->ij", count, center_xtwy_grid, optimize=True
        )
        current_center = np.linalg.solve(
            current_center_xtwx, current_center_xtwy
        )
        current_weight = float(count @ weight_grid)
        current_average_basis = (
            count @ basis_weight_grid
        ) / current_weight
        current_effects = window_effects(current_beta)
        current_regional_mean = np.asarray(
            [
                current_average_basis @ (current_center[:, 0] + effect[:, 0])
                for effect in current_effects
            ]
        )
        current_regional_variance = np.asarray(
            [
                current_average_basis @ (current_center[:, 1] + effect[:, 1])
                for effect in current_effects
            ]
        )
        current_regional_log_stability = np.log(current_regional_mean) - 0.5 * np.log(
            current_regional_variance
        )
        regional_window_bootstrap[draw] = current_regional_log_stability
        regional_bootstrap[draw] = (
            current_regional_log_stability[-1]
            - current_regional_log_stability[0]
        )
        regional_slope_bootstrap[draw] = np.polyfit(
            TIME, current_regional_log_stability, 1
        )[0]
        current_node_mean = np.stack(
            [current_center[:, 0] + effect[:, 0] for effect in current_effects]
        )
        current_node_variance = np.stack(
            [current_center[:, 1] + effect[:, 1] for effect in current_effects]
        )
        current_dense_mean = current_node_mean @ dense_basis.T
        current_dense_variance = current_node_variance @ dense_basis.T
        current_node_log_stability = np.log(current_node_mean) - 0.5 * np.log(
            current_node_variance
        )
        current_dense_log_stability = np.log(
            current_dense_mean
        ) - 0.5 * np.log(current_dense_variance)
        node_window_bootstrap[draw] = current_node_log_stability
        dense_window_bootstrap[draw] = current_dense_log_stability
        node_bootstrap[draw] = (
            current_node_log_stability[-1] - current_node_log_stability[0]
        )
        dense_bootstrap[draw] = (
            current_dense_log_stability[-1] - current_dense_log_stability[0]
        )

    node_change = stability_log_difference(
        center_beta[:, 0] + effects[0, :, 0],
        center_beta[:, 1] + effects[0, :, 1],
        center_beta[:, 0] + effects[-1, :, 0],
        center_beta[:, 1] + effects[-1, :, 1],
    )
    contrast = np.column_stack(
        (
            np.eye(len(DISTANCE_NODES_M) - 1),
            -np.ones(len(DISTANCE_NODES_M) - 1),
        )
    )
    node_difference = contrast @ node_change
    bootstrap_difference = node_bootstrap @ contrast.T
    difference_covariance = np.cov(bootstrap_difference, rowvar=False, ddof=1)
    degrees = int(np.linalg.matrix_rank(difference_covariance))
    statistic = float(
        node_difference @ np.linalg.pinv(difference_covariance) @ node_difference
    )
    heterogeneity_p = float(chi2.sf(statistic, degrees))

    summary_rows = []
    for index, center_year in enumerate(WINDOW_CENTERS):
        current_mean = float(np.average(window_mean[:, index], weights=pixel_weight))
        current_variance = float(
            np.average(window_variance[:, index], weights=pixel_weight)
        )
        summary_rows.append(
            {
                "window_start_year": int(YEARS[WINDOW_STARTS[index]]),
                "window_end_year": int(
                    YEARS[WINDOW_STARTS[index] + WINDOW_N - 1]
                ),
                "window_center_year": int(center_year),
                "weighted_mean_nirv": current_mean,
                "weighted_within_pixel_temporal_variance": current_variance,
                "weighted_within_pixel_temporal_sd": np.sqrt(current_variance),
                "regional_temporal_stability": current_mean
                / np.sqrt(current_variance),
                "quality_adjusted_mean_nirv": regional_mean[index],
                "quality_adjusted_temporal_variance": regional_variance[index],
                "quality_adjusted_temporal_sd": np.sqrt(regional_variance[index]),
                "quality_adjusted_temporal_stability": np.exp(
                    regional_log_stability[index]
                ),
                "quality_adjusted_stability_lower_95": np.exp(
                    np.quantile(regional_window_bootstrap[:, index], 0.025)
                ),
                "quality_adjusted_stability_upper_95": np.exp(
                    np.quantile(regional_window_bootstrap[:, index], 0.975)
                ),
                "pixel_n": len(distance),
                "grid_n": grid_n,
            }
        )
    summaries = pd.DataFrame(summary_rows)

    adjacent_rows = []
    adjacent_log_change = np.diff(regional_log_stability)
    adjacent_bootstrap = np.diff(regional_window_bootstrap, axis=1)
    for index, estimate in enumerate(adjacent_log_change):
        current_draw = adjacent_bootstrap[:, index]
        lower, upper = np.quantile(current_draw, [0.025, 0.975])
        adjacent_rows.append(
            {
                "from_window": f"{YEARS[WINDOW_STARTS[index]]}--{YEARS[WINDOW_STARTS[index] + WINDOW_N - 1]}",
                "to_window": f"{YEARS[WINDOW_STARTS[index + 1]]}--{YEARS[WINDOW_STARTS[index + 1] + WINDOW_N - 1]}",
                "estimate_log_change": estimate,
                "estimate_percent_change": 100.0 * np.expm1(estimate),
                "lower_95_percent_change": 100.0 * np.expm1(lower),
                "upper_95_percent_change": 100.0 * np.expm1(upper),
                "p_value": p_from_bootstrap(current_draw),
                "point_estimate_direction": (
                    "down" if estimate < 0.0 else "up" if estimate > 0.0 else "flat"
                ),
            }
        )
    adjacent_changes = pd.DataFrame(adjacent_rows)

    node_mean = np.stack(
        [center_beta[:, 0] + current[:, 0] for current in effects]
    )
    node_variance = np.stack(
        [center_beta[:, 1] + current[:, 1] for current in effects]
    )
    node_log_stability = np.log(node_mean) - 0.5 * np.log(node_variance)
    node_stability = np.exp(node_log_stability)
    node_window_lower = np.exp(
        np.quantile(node_window_bootstrap, 0.025, axis=0)
    )
    node_window_upper = np.exp(
        np.quantile(node_window_bootstrap, 0.975, axis=0)
    )
    gradient_log_change = node_log_stability[:, 0] - node_log_stability[:, -1]
    gradient_bootstrap = (
        node_window_bootstrap[:, :, 0] - node_window_bootstrap[:, :, -1]
    )
    gradient_rows = []
    for index, center_year in enumerate(WINDOW_CENTERS):
        current_gradient_draw = gradient_bootstrap[:, index]
        gradient_lower, gradient_upper = np.quantile(
            current_gradient_draw, [0.025, 0.975]
        )
        row = {
            "window_start_year": int(YEARS[WINDOW_STARTS[index]]),
            "window_end_year": int(YEARS[WINDOW_STARTS[index] + WINDOW_N - 1]),
            "window_center_year": int(center_year),
            "stability_30m_vs_480m_percent": 100.0
            * np.expm1(gradient_log_change[index]),
            "stability_30m_vs_480m_lower_95": 100.0
            * np.expm1(gradient_lower),
            "stability_30m_vs_480m_upper_95": 100.0
            * np.expm1(gradient_upper),
            "stability_30m_vs_480m_p_value": p_from_bootstrap(
                current_gradient_draw
            ),
        }
        for node_index, node in enumerate(DISTANCE_NODES_M):
            label = int(node)
            row[f"stability_{label}m"] = node_stability[index, node_index]
            row[f"stability_{label}m_lower_95"] = node_window_lower[
                index, node_index
            ]
            row[f"stability_{label}m_upper_95"] = node_window_upper[
                index, node_index
            ]
        gradient_rows.append(row)
    gradients = pd.DataFrame(gradient_rows)

    distance_window_lower = np.exp(
        np.quantile(dense_window_bootstrap, 0.025, axis=0)
    )
    distance_window_upper = np.exp(
        np.quantile(dense_window_bootstrap, 0.975, axis=0)
    )
    distance_window_rows = []
    for window_index, center_year in enumerate(WINDOW_CENTERS):
        for distance_index, current_distance in enumerate(DENSE_DISTANCE_M):
            distance_window_rows.append(
                {
                    "window_start_year": int(YEARS[WINDOW_STARTS[window_index]]),
                    "window_end_year": int(
                        YEARS[WINDOW_STARTS[window_index] + WINDOW_N - 1]
                    ),
                    "window_center_year": int(center_year),
                    "distance_m": current_distance,
                    "quality_adjusted_mean_nirv": dense_mean[
                        window_index, distance_index
                    ],
                    "quality_adjusted_temporal_variance": dense_variance[
                        window_index, distance_index
                    ],
                    "quality_adjusted_temporal_stability": np.exp(
                        np.log(dense_mean[window_index, distance_index])
                        - 0.5
                        * np.log(dense_variance[window_index, distance_index])
                    ),
                    "stability_lower_95": distance_window_lower[
                        window_index, distance_index
                    ],
                    "stability_upper_95": distance_window_upper[
                        window_index, distance_index
                    ],
                }
            )
    distance_windows = pd.DataFrame(distance_window_rows)

    curve_rows = []
    dense_lower = np.quantile(dense_bootstrap, 0.025, axis=0)
    dense_upper = np.quantile(dense_bootstrap, 0.975, axis=0)
    for index, current_distance in enumerate(DENSE_DISTANCE_M):
        current_mean_first = dense_mean[0, index]
        current_mean_last = dense_mean[-1, index]
        current_variance_first = dense_variance[0, index]
        current_variance_last = dense_variance[-1, index]
        current_stability_first = current_mean_first / np.sqrt(
            current_variance_first
        )
        current_stability_last = current_mean_last / np.sqrt(
            current_variance_last
        )
        curve_rows.append(
            {
                "distance_m": current_distance,
                "first_window_start_year": int(YEARS[WINDOW_STARTS[0]]),
                "first_window_end_year": int(
                    YEARS[WINDOW_STARTS[0] + WINDOW_N - 1]
                ),
                "last_window_start_year": int(YEARS[WINDOW_STARTS[-1]]),
                "last_window_end_year": int(
                    YEARS[WINDOW_STARTS[-1] + WINDOW_N - 1]
                ),
                "mean_nirv_first_window": current_mean_first,
                "mean_nirv_last_window": current_mean_last,
                "temporal_variance_first_window": current_variance_first,
                "temporal_variance_last_window": current_variance_last,
                "stability_first_window": current_stability_first,
                "stability_last_window": current_stability_last,
                "stability_log_change_first_to_last": log_change[index],
                "stability_percent_change_first_to_last": 100.0
                * np.expm1(log_change[index]),
                "stability_percent_change_lower_95": 100.0
                * np.expm1(dense_lower[index]),
                "stability_percent_change_upper_95": 100.0
                * np.expm1(dense_upper[index]),
            }
        )
    curves = pd.DataFrame(curve_rows)

    regional_lower, regional_upper = np.quantile(
        regional_bootstrap, [0.025, 0.975]
    )
    slope_lower, slope_upper = np.quantile(
        regional_slope_bootstrap, [0.025, 0.975]
    )
    regional_tests = pd.DataFrame(
        [
            {
                "test": "regional_stability_change_first_to_last_window",
                "comparison": (
                    f"{YEARS[WINDOW_STARTS[0]]}--{YEARS[WINDOW_STARTS[0] + WINDOW_N - 1]} "
                    f"window versus {YEARS[WINDOW_STARTS[-1]]}--{YEARS[WINDOW_STARTS[-1] + WINDOW_N - 1]} window"
                ),
                "estimate_log_change": regional_log_change,
                "estimate_percent_change": 100.0 * np.expm1(regional_log_change),
                "lower_95_percent_change": 100.0 * np.expm1(regional_lower),
                "upper_95_percent_change": 100.0 * np.expm1(regional_upper),
                "p_value": p_from_bootstrap(regional_bootstrap),
                "wald_chi_square": np.nan,
                "wald_df": np.nan,
            },
            {
                "test": "regional_linear_log_stability_trend",
                "comparison": (
                    f"linear change across rolling-window centers "
                    f"{WINDOW_CENTERS[0]}--{WINDOW_CENTERS[-1]}"
                ),
                "estimate_log_change": regional_linear_slope,
                "estimate_percent_change": 100.0 * np.expm1(regional_linear_slope),
                "lower_95_percent_change": 100.0 * np.expm1(slope_lower),
                "upper_95_percent_change": 100.0 * np.expm1(slope_upper),
                "p_value": p_from_bootstrap(regional_slope_bootstrap),
                "wald_chi_square": np.nan,
                "wald_df": np.nan,
            },
            {
                "test": "distance_heterogeneity_in_stability_change",
                "comparison": "equality of first-to-last log change at 30, 60, 120, 240, and 480 m",
                "estimate_log_change": np.nan,
                "estimate_percent_change": np.nan,
                "lower_95_percent_change": np.nan,
                "upper_95_percent_change": np.nan,
                "p_value": heterogeneity_p,
                "wald_chi_square": statistic,
                "wald_df": degrees,
            },
        ]
    )

    support_rows = []
    for lower, upper in zip(DISTANCE_NODES_M[:-1], DISTANCE_NODES_M[1:]):
        keep = (distance >= lower) & (
            distance < upper if upper < 480.0 else distance <= upper
        )
        support_rows.append(
            {
                "distance_lower_m": lower,
                "distance_upper_m": upper,
                "pixel_n": int(keep.sum()),
                "grid_n": int(np.unique(grid[keep]).size),
                "base_weight_sum": float(pixel_weight[keep].sum()),
            }
        )
    support = pd.DataFrame(support_rows)

    temporary_root = Path(tempfile.mkdtemp(prefix="regional_stability_trend_"))
    temporary_output = temporary_root / OUTPUT.name
    temporary_output.mkdir()
    try:
        summaries.to_csv(
            temporary_output / "rolling_window_regional_stability.csv", index=False
        )
        adjacent_changes.to_csv(
            temporary_output / "regional_adjacent_window_changes.csv", index=False
        )
        gradients.to_csv(
            temporary_output / "distance_gradient_by_window.csv", index=False
        )
        distance_windows.to_csv(
            temporary_output / "distance_stability_by_window.csv", index=False
        )
        curves.to_csv(
            temporary_output / "distance_stability_time_change_curves.csv",
            index=False,
        )
        regional_tests.to_csv(
            temporary_output / "stability_time_change_tests.csv", index=False
        )
        support.to_csv(
            temporary_output / "balanced_panel_distance_support.csv", index=False
        )
        metadata = {
            "status": "COMPLETE_REGIONAL_STABILITY_TIME_TREND",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(INPUT.relative_to(ROOT)),
            "years": YEARS.tolist(),
            "rolling_window_years": WINDOW_N,
            "window_centers": WINDOW_CENTERS.tolist(),
            "formal_distance_range_m": [30, 480],
            "balanced_pixel_n": int(len(distance)),
            "grid_n": int(grid_n),
            "eligibility": "formal 30--480 m pixels with valid NIRv, distance, and at least one valid scene in all 13 years",
            "regional_estimand": f"area-weighted mean NIRv divided by the square root of pooled within-pixel temporal variance in successive {WINDOW_N}-year windows",
            "time_comparison": (
                f"first {YEARS[WINDOW_STARTS[0]]}--{YEARS[WINDOW_STARTS[0] + WINDOW_N - 1]} window "
                f"versus last {YEARS[WINDOW_STARTS[-1]]}--{YEARS[WINDOW_STARTS[-1] + WINDOW_N - 1]} window"
            ),
            "time_model": f"pixel-fixed-effect {len(WINDOW_CENTERS)}-window factor with the first-to-last contrast primary and a linear summary secondary",
            "distance_model": "window-by-piecewise-linear log-distance basis at 30, 60, 120, 240, and 480 m",
            "observation_quality_controls": [
                "within-pixel rolling-window fraction with one valid scene",
                "within-pixel rolling-window fraction with two to three valid scenes",
            ],
            "weights": "frozen base_weight",
            "inference": f"{BOOTSTRAP_DRAWS} spatial 50-km-grid cluster bootstrap draws",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "regional_percent_change_first_to_last": float(
                100.0 * np.expm1(regional_log_change)
            ),
            "regional_linear_percent_change_per_window_center_year": float(
                100.0 * np.expm1(regional_linear_slope)
            ),
            "regional_adjacent_step_n": int(len(adjacent_log_change)),
            "regional_downward_adjacent_step_n": int(
                np.sum(adjacent_log_change < 0.0)
            ),
            "regional_all_point_estimate_steps_down": bool(
                np.all(adjacent_log_change < 0.0)
            ),
            "cluster_bootstrap_fraction_all_steps_down": float(
                np.mean(np.all(adjacent_bootstrap < 0.0, axis=1))
            ),
            "distance_heterogeneity_p_value": heterogeneity_p,
        }
        (temporary_output / "analysis_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary_output.rename(OUTPUT)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    print(f"COMPLETE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-years", type=int, default=WINDOW_N)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
    )
    arguments = parser.parse_args()
    configured_output = arguments.output
    if not configured_output.is_absolute():
        configured_output = ROOT / configured_output
    configure(arguments.window_years, configured_output)
    main()
