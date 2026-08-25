#!/usr/bin/env python3
"""Check climate interactions after allowing nonlinear single-variable effects."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "analysis_inputs" / "annual_climate_state"
RESULTS = ROOT / "outputs" / "core_results" / "02_environmental_states"
LEGACY_STATES = (
    ROOT
    / "outputs"
    / "archive"
    / "legacy_two_dimensional_states"
    / "02_environmental_states"
)
OUTPUT = RESULTS / "nonlinear_main_effect_check"

YEARS = tuple(range(2013, 2026))
NODES_M = np.asarray([30.0, 60.0, 120.0, 240.0, 480.0])
FREE_NODE_N = 4
DENSE_DISTANCE_M = np.geomspace(30.0, 480.0, 97)
VARIABLES = (
    "shortwave",
    "temperature",
    "vpd",
    "rootzone_soil_moisture",
)
BASE_COMPONENTS = (
    "shortwave",
    "temperature",
    "vpd",
    "rootzone_soil_moisture",
    "shortwave_x_temperature",
    "shortwave_x_vpd",
    "shortwave_x_rootzone_soil_moisture",
    "temperature_x_vpd",
    "temperature_x_rootzone_soil_moisture",
    "vpd_x_rootzone_soil_moisture",
)
NONLINEAR_COMPONENTS = tuple(
    f"{variable}_nonlinear_{basis}"
    for variable in VARIABLES
    for basis in (1, 2)
)
EXTENDED_COMPONENTS = BASE_COMPONENTS + NONLINEAR_COMPONENTS
OUTCOMES = (
    "positive_probability",
    "positive_conditional_second_moment",
    "negative_conditional_second_moment",
)
OUTCOME_LABELS = {
    "positive_probability": "正偏离概率",
    "positive_conditional_second_moment": "正偏离条件平方幅度",
    "negative_conditional_second_moment": "负偏离条件平方幅度",
}
COMPONENT_LABELS = {
    "shortwave": "短波",
    "temperature": "温度",
    "vpd": "VPD",
    "rootzone_soil_moisture": "根区土壤水分",
    "shortwave_x_temperature": "短波×温度",
    "shortwave_x_vpd": "短波×VPD",
    "shortwave_x_rootzone_soil_moisture": "短波×根区土壤水分",
    "temperature_x_vpd": "温度×VPD",
    "temperature_x_rootzone_soil_moisture": "温度×根区土壤水分",
    "vpd_x_rootzone_soil_moisture": "VPD×根区土壤水分",
}
PAIR_VARIABLES = {
    "shortwave_x_temperature": ("shortwave", "temperature"),
    "shortwave_x_vpd": ("shortwave", "vpd"),
    "shortwave_x_rootzone_soil_moisture": (
        "shortwave",
        "rootzone_soil_moisture",
    ),
    "temperature_x_vpd": ("temperature", "vpd"),
    "temperature_x_rootzone_soil_moisture": (
        "temperature",
        "rootzone_soil_moisture",
    ),
    "vpd_x_rootzone_soil_moisture": (
        "vpd",
        "rootzone_soil_moisture",
    ),
}
CANDIDATE_STATES = (
    (
        "temperature_x_rootzone_soil_moisture",
        "high_temperature__high_rootzone_soil_moisture",
        "positive_conditional_second_moment",
        1,
    ),
    (
        "temperature_x_vpd",
        "high_temperature__high_vpd",
        "positive_probability",
        -1,
    ),
    (
        "temperature_x_vpd",
        "high_temperature__high_vpd",
        "negative_conditional_second_moment",
        1,
    ),
)


def group_sum(value: np.ndarray, group: np.ndarray, group_n: int) -> np.ndarray:
    return np.bincount(group, weights=value, minlength=group_n).astype(np.float64)


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


def build_long_arrays(sample: pd.DataFrame) -> dict[str, np.ndarray]:
    grid, grid_levels = pd.factorize(sample["grid_id"], sort=True)
    pixel_weight = sample["base_weight"].to_numpy(np.float64, copy=True)
    pixel_weight /= pixel_weight.mean()
    between_x = np.log10(sample["d_between_m"].to_numpy(np.float64, copy=True))
    pixel = np.arange(len(sample), dtype=np.int32)
    parts = {
        key: []
        for key in (
            "y",
            "x_within",
            "weight",
            "scene",
            "year",
            "grid",
            "pixel",
        )
    }
    for year_index, year in enumerate(YEARS):
        distance = sample[f"distance_m_{year}"].to_numpy(np.float64, copy=True)
        nirv = sample[f"nirv_{year}"].to_numpy(np.float64, copy=True)
        scene = sample[f"valid_scene_n_{year}"].to_numpy(np.float64, copy=True)
        keep = (
            np.isfinite(distance)
            & (distance > 0)
            & np.isfinite(nirv)
            & (scene >= 1)
        )
        parts["y"].append(nirv[keep])
        parts["x_within"].append(np.log10(distance[keep]) - between_x[keep])
        parts["weight"].append(pixel_weight[keep])
        parts["scene"].append(
            np.where(scene[keep] == 1, 0, np.where(scene[keep] <= 3, 1, 2)).astype(
                np.int32
            )
        )
        parts["year"].append(np.full(keep.sum(), year_index, dtype=np.int32))
        parts["grid"].append(grid[keep].astype(np.int32))
        parts["pixel"].append(pixel[keep])
    output = {
        name: np.ascontiguousarray(
            np.concatenate(values),
            dtype=(
                np.int32
                if name in {"scene", "year", "grid", "pixel"}
                else np.float64
            ),
        )
        for name, values in parts.items()
    }
    output["grid_levels"] = np.asarray(grid_levels, dtype=object)
    output["pixel_grid"] = grid.astype(np.int32)
    output["pixel_between_x"] = between_x
    return output


def mean_residual(
    arrays: dict[str, np.ndarray], pixel_n: int
) -> tuple[np.ndarray, dict[str, float]]:
    y = arrays["y"]
    x_within = arrays["x_within"]
    weight = arrays["weight"]
    year = arrays["year"]
    pixel = arrays["pixel"]
    pixel_weight = group_sum(weight, pixel, pixel_n)
    year_weight = group_sum(weight, year, len(YEARS))
    pixel_intercept = group_sum(weight * y, pixel, pixel_n) / pixel_weight
    year_effect = np.zeros(len(YEARS), dtype=np.float64)
    beta_within = 0.0
    for iteration in range(1, 61):
        previous_year = year_effect.copy()
        previous_beta = beta_within
        pixel_intercept = group_sum(
            weight * (y - year_effect[year] - beta_within * x_within),
            pixel,
            pixel_n,
        ) / pixel_weight
        year_effect = group_sum(
            weight * (y - pixel_intercept[pixel] - beta_within * x_within),
            year,
            len(YEARS),
        ) / year_weight
        center = np.average(year_effect, weights=year_weight)
        year_effect -= center
        pixel_intercept += center
        partial = y - pixel_intercept[pixel] - year_effect[year]
        beta_within = float(
            np.sum(weight * x_within * partial)
            / np.sum(weight * np.square(x_within))
        )
        if max(
            abs(beta_within - previous_beta),
            np.max(np.abs(year_effect - previous_year)),
        ) < 1e-10:
            break
    else:
        raise RuntimeError("mean residualization did not converge")
    residual = (
        y
        - pixel_intercept[pixel]
        - year_effect[year]
        - beta_within * x_within
    )
    return residual, {
        "iterations": iteration,
        "within_effect_per_doubling": beta_within * np.log10(2.0),
    }


def restricted_cubic_columns(
    value: np.ndarray, knots: np.ndarray
) -> np.ndarray:
    if len(knots) != 4 or np.any(np.diff(knots) <= 0):
        raise RuntimeError("four ordered restricted-cubic-spline knots are required")

    def positive_cube(knot: float) -> np.ndarray:
        return np.maximum(value - knot, 0.0) ** 3

    denominator = knots[3] - knots[2]
    scale = (knots[3] - knots[0]) ** 2
    columns = []
    for knot in knots[:2]:
        column = (
            positive_cube(knot)
            - positive_cube(knots[2]) * (knots[3] - knot) / denominator
            + positive_cube(knots[3]) * (knots[2] - knot) / denominator
        ) / scale
        columns.append(column)
    return np.column_stack(columns)


def align_climate(
    climate: pd.DataFrame,
    grid_levels: np.ndarray,
    validate_frozen_definitions: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    expected = pd.MultiIndex.from_product(
        [grid_levels.astype(str), YEARS], names=["grid_id", "year"]
    )
    aligned = climate.set_index(["grid_id", "year"]).reindex(expected)
    if aligned.isna().any().any():
        raise RuntimeError("climate table cannot be aligned to formal grid-years")
    raw = np.column_stack(
        [aligned[f"{variable}_z"].to_numpy(np.float64) for variable in VARIABLES]
    ).reshape(len(grid_levels), len(YEARS), len(VARIABLES))

    base_values: list[np.ndarray] = []
    definition_rows = []
    for variable_index, variable in enumerate(VARIABLES):
        value = raw[:, :, variable_index]
        centered = value - value.mean(axis=1, keepdims=True)
        scale = float(centered.reshape(-1).std(ddof=1))
        base_values.append(centered / scale)
        definition_rows.append(
            {
                "component": variable,
                "component_type": "linear_main",
                "source_variable": variable,
                "pooled_sd_before_scaling": scale,
                "knot_05": np.nan,
                "knot_35": np.nan,
                "knot_65": np.nan,
                "knot_95": np.nan,
            }
        )

    variable_index = {variable: index for index, variable in enumerate(VARIABLES)}
    for component in BASE_COMPONENTS[4:]:
        left, right = PAIR_VARIABLES[component]
        value = (
            raw[:, :, variable_index[left]] * raw[:, :, variable_index[right]]
        )
        value -= value.mean(axis=1, keepdims=True)
        scale = float(value.reshape(-1).std(ddof=1))
        base_values.append(value / scale)
        definition_rows.append(
            {
                "component": component,
                "component_type": "pairwise",
                "source_variable": f"{left};{right}",
                "pooled_sd_before_scaling": scale,
                "knot_05": np.nan,
                "knot_35": np.nan,
                "knot_65": np.nan,
                "knot_95": np.nan,
            }
        )
    base = np.stack(base_values, axis=2)

    nonlinear_values = []
    knot_registry: dict[str, np.ndarray] = {}
    for variable_index_value, variable in enumerate(VARIABLES):
        value = raw[:, :, variable_index_value]
        knots = np.quantile(value.reshape(-1), [0.05, 0.35, 0.65, 0.95])
        knot_registry[variable] = knots
        nonlinear = restricted_cubic_columns(value.reshape(-1), knots).reshape(
            len(grid_levels), len(YEARS), 2
        )
        nonlinear -= nonlinear.mean(axis=1, keepdims=True)
        orthogonal = [base[:, :, variable_index_value].reshape(-1)]
        for basis_index in range(2):
            current = nonlinear[:, :, basis_index].reshape(-1)
            design = np.column_stack(orthogonal)
            current -= design @ np.linalg.lstsq(design, current, rcond=None)[0]
            current = current.reshape(len(grid_levels), len(YEARS))
            current -= current.mean(axis=1, keepdims=True)
            scale = float(current.reshape(-1).std(ddof=1))
            current /= scale
            nonlinear_values.append(current)
            orthogonal.append(current.reshape(-1))
            definition_rows.append(
                {
                    "component": f"{variable}_nonlinear_{basis_index + 1}",
                    "component_type": "restricted_cubic_nonlinear_main",
                    "source_variable": variable,
                    "pooled_sd_before_scaling": scale,
                    "knot_05": knots[0],
                    "knot_35": knots[1],
                    "knot_65": knots[2],
                    "knot_95": knots[3],
                }
            )
    extended = np.concatenate(
        [base, np.stack(nonlinear_values, axis=2)], axis=2
    )
    if np.max(np.abs(extended.mean(axis=1))) > 1e-12:
        raise RuntimeError("one or more climate components are not within-grid centered")
    if np.max(np.abs(extended.reshape(-1, extended.shape[2]).std(axis=0, ddof=1) - 1)) > 1e-10:
        raise RuntimeError("one or more climate components are not standardized")

    if validate_frozen_definitions:
        frozen_definitions = pd.read_csv(
            RESULTS / "climate_component_definitions.csv"
        )
        reproduced_scale = pd.DataFrame(definition_rows[: len(BASE_COMPONENTS)])[
            "pooled_sd_before_scaling"
        ].to_numpy()
        if np.max(
            np.abs(
                reproduced_scale
                - frozen_definitions[
                    "pooled_grid_year_sd_before_scaling"
                ].to_numpy()
            )
        ) > 1e-12:
            raise RuntimeError("base climate component scales were not reproduced")
    return raw, extended, pd.DataFrame(definition_rows), knot_registry


def absorption_setup(
    weight: np.ndarray,
    pixel: np.ndarray,
    grid_year: np.ndarray,
    pixel_n: int,
    grid_year_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        group_sum(weight, pixel, pixel_n),
        group_sum(weight, grid_year, grid_year_n),
    )


def absorb(
    value: np.ndarray,
    weight: np.ndarray,
    pixel: np.ndarray,
    grid_year: np.ndarray,
    pixel_denominator: np.ndarray,
    grid_year_denominator: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    result = np.asarray(value, dtype=np.float64).copy()
    for iteration in range(1, 151):
        before = result.copy()
        pixel_sum = group_sum(weight * result, pixel, len(pixel_denominator))
        pixel_mean = np.divide(
            pixel_sum,
            pixel_denominator,
            out=np.zeros_like(pixel_sum),
            where=pixel_denominator > 0,
        )
        result -= pixel_mean[pixel]
        grid_year_sum = group_sum(
            weight * result, grid_year, len(grid_year_denominator)
        )
        grid_year_mean = np.divide(
            grid_year_sum,
            grid_year_denominator,
            out=np.zeros_like(grid_year_sum),
            where=grid_year_denominator > 0,
        )
        result -= grid_year_mean[grid_year]
        final_change = float(np.max(np.abs(result - before)))
        if final_change <= 1e-9:
            return result, iteration, final_change
    raise RuntimeError(f"two-way absorption did not converge: {final_change:.3e}")


def cross_products(
    design: np.memmap,
    outcome: np.ndarray,
    weight: np.ndarray,
    chunk_size: int = 150_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    parameter_n = design.shape[1]
    xtwx = np.zeros((parameter_n, parameter_n), dtype=np.float64)
    xtwy = np.zeros(parameter_n, dtype=np.float64)
    yty = 0.0
    for start in range(0, len(weight), chunk_size):
        stop = min(start + chunk_size, len(weight))
        x = np.asarray(design[start:stop], dtype=np.float64)
        y = outcome[start:stop]
        w = weight[start:stop]
        xtwx += x.T @ (x * w[:, None])
        xtwy += x.T @ (w * y)
        yty += float(np.sum(w * np.square(y)))
    return xtwx, xtwy, yty


def parameter_names(components: tuple[str, ...]) -> list[str]:
    names = ["scene_1_vs_ge4", "scene_2to3_vs_ge4"]
    names.extend(
        f"{component}_x_distance_{int(distance)}m_vs480"
        for component in components
        for distance in NODES_M[:FREE_NODE_N]
    )
    return names


def fit_outcome(
    outcome_name: str,
    indexes: np.ndarray,
    outcome_raw: np.ndarray,
    arrays: dict[str, np.ndarray],
    pixel_basis: np.ndarray,
    climate_components: np.ndarray,
    temporary_root: Path,
    pixel_n: int,
    grid_year_n: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, float]]:
    weight = arrays["weight"][indexes]
    pixel = arrays["pixel"][indexes]
    grid = arrays["grid"][indexes]
    year = arrays["year"][indexes]
    grid_year = grid * len(YEARS) + year
    scene = arrays["scene"][indexes]
    pixel_denominator, grid_year_denominator = absorption_setup(
        weight, pixel, grid_year, pixel_n, grid_year_n
    )
    extended_names = parameter_names(EXTENDED_COMPONENTS)
    benchmark_n = len(parameter_names(BASE_COMPONENTS))
    design_path = temporary_root / f"{outcome_name}_design.dat"
    design = np.memmap(
        design_path,
        dtype=np.float64,
        mode="w+",
        shape=(len(indexes), len(extended_names)),
    )
    maximum_iterations = 0
    maximum_final_change = 0.0
    for column, raw in enumerate(((scene == 0), (scene == 1))):
        transformed, iterations, final_change = absorb(
            raw.astype(np.float64),
            weight,
            pixel,
            grid_year,
            pixel_denominator,
            grid_year_denominator,
        )
        design[:, column] = transformed
        maximum_iterations = max(maximum_iterations, iterations)
        maximum_final_change = max(maximum_final_change, final_change)

    column = 2
    for component_index, component in enumerate(EXTENDED_COMPONENTS):
        climate_value = climate_components[grid, year, component_index]
        for node_index in range(FREE_NODE_N):
            raw = climate_value * pixel_basis[pixel, node_index]
            transformed, iterations, final_change = absorb(
                raw,
                weight,
                pixel,
                grid_year,
                pixel_denominator,
                grid_year_denominator,
            )
            design[:, column] = transformed
            column += 1
            maximum_iterations = max(maximum_iterations, iterations)
            maximum_final_change = max(maximum_final_change, final_change)
        print(f"  {outcome_name}: {component}", flush=True)
    design.flush()

    outcome, iterations, final_change = absorb(
        outcome_raw[indexes],
        weight,
        pixel,
        grid_year,
        pixel_denominator,
        grid_year_denominator,
    )
    maximum_iterations = max(maximum_iterations, iterations)
    maximum_final_change = max(maximum_final_change, final_change)
    xtwx_extended, xtwy_extended, yty = cross_products(design, outcome, weight)
    xtwx_benchmark = xtwx_extended[:benchmark_n, :benchmark_n]
    xtwy_benchmark = xtwy_extended[:benchmark_n]
    if np.linalg.matrix_rank(xtwx_benchmark) != benchmark_n:
        raise RuntimeError(f"benchmark design is rank deficient: {outcome_name}")
    if np.linalg.matrix_rank(xtwx_extended) != len(extended_names):
        raise RuntimeError(f"extended design is rank deficient: {outcome_name}")
    inverse_benchmark = np.linalg.inv(xtwx_benchmark)
    inverse_extended = np.linalg.inv(xtwx_extended)
    beta_benchmark = inverse_benchmark @ xtwy_benchmark
    beta_extended = inverse_extended @ xtwy_extended

    score_benchmark = np.zeros((len(arrays["grid_levels"]), benchmark_n))
    score_extended = np.zeros((len(arrays["grid_levels"]), len(extended_names)))
    sse_benchmark = 0.0
    sse_extended = 0.0
    for start in range(0, len(weight), 150_000):
        stop = min(start + 150_000, len(weight))
        x = np.asarray(design[start:stop], dtype=np.float64)
        y = outcome[start:stop]
        w = weight[start:stop]
        cluster = grid[start:stop]
        residual_benchmark = y - x[:, :benchmark_n] @ beta_benchmark
        residual_extended = y - x @ beta_extended
        sse_benchmark += float(np.sum(w * np.square(residual_benchmark)))
        sse_extended += float(np.sum(w * np.square(residual_extended)))
        for parameter_index in range(benchmark_n):
            score_benchmark[:, parameter_index] += np.bincount(
                cluster,
                weights=(
                    x[:, parameter_index] * w * residual_benchmark
                ),
                minlength=len(arrays["grid_levels"]),
            )
        for parameter_index in range(len(extended_names)):
            score_extended[:, parameter_index] += np.bincount(
                cluster,
                weights=x[:, parameter_index] * w * residual_extended,
                minlength=len(arrays["grid_levels"]),
            )

    design._mmap.close()
    del design
    design_path.unlink()
    benchmark = {
        "beta": beta_benchmark,
        "inverse": inverse_benchmark,
        "score": score_benchmark,
    }
    extended = {
        "beta": beta_extended,
        "inverse": inverse_extended,
        "score": score_extended,
    }
    diagnostics = {
        "observation_n": int(len(indexes)),
        "maximum_absorption_iterations": int(maximum_iterations),
        "maximum_absorption_final_change": maximum_final_change,
        "benchmark_condition_number": float(np.linalg.cond(xtwx_benchmark)),
        "extended_condition_number": float(np.linalg.cond(xtwx_extended)),
        "benchmark_absorbed_weighted_r2": float(1 - sse_benchmark / yty),
        "extended_absorbed_weighted_r2": float(1 - sse_extended / yty),
    }
    return benchmark, extended, diagnostics


def assemble_joint(
    fitted: list[dict[str, object]],
    observation_n: int,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    parameter_n = len(fitted[0]["beta"])
    total_parameter_n = parameter_n * len(fitted)
    beta = np.concatenate([current["beta"] for current in fitted])
    bread = np.zeros((total_parameter_n, total_parameter_n))
    scores = np.zeros((grid_n, total_parameter_n))
    for outcome_index, current in enumerate(fitted):
        start = outcome_index * parameter_n
        stop = start + parameter_n
        bread[start:stop, start:stop] = current["inverse"]
        scores[:, start:stop] = current["score"]
    correction = grid_n / (grid_n - 1.0)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return beta, covariance, correction


def wald(
    beta: np.ndarray, covariance: np.ndarray, indices: np.ndarray
) -> tuple[float, int, float]:
    estimate = beta[indices]
    current_covariance = covariance[np.ix_(indices, indices)]
    degrees = int(np.linalg.matrix_rank(current_covariance))
    statistic = float(estimate @ np.linalg.pinv(current_covariance) @ estimate)
    return statistic, degrees, float(chi2.sf(statistic, degrees))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * sorted_p
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def parameter_index(
    names: list[str], component: str, distance_m: int
) -> int:
    return names.index(f"{component}_x_distance_{distance_m}m_vs480")


def component_tests(
    beta: np.ndarray,
    covariance: np.ndarray,
    names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_n = len(names)
    rows = []
    nonlinear_rows = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        offset = outcome_index * parameter_n
        for variable in VARIABLES:
            components = (
                variable,
                f"{variable}_nonlinear_1",
                f"{variable}_nonlinear_2",
            )
            indices = np.asarray(
                [
                    offset + parameter_index(names, component, int(distance))
                    for component in components
                    for distance in NODES_M[:FREE_NODE_N]
                ]
            )
            statistic, degrees, p_value = wald(beta, covariance, indices)
            rows.append(
                {
                    "outcome": outcome,
                    "component": variable,
                    "component_type": "flexible_main",
                    "wald_chi_square": statistic,
                    "wald_df": degrees,
                    "p_value": p_value,
                }
            )
            nonlinear_indices = np.asarray(
                [
                    offset + parameter_index(names, component, int(distance))
                    for component in components[1:]
                    for distance in NODES_M[:FREE_NODE_N]
                ]
            )
            statistic, degrees, p_value = wald(
                beta, covariance, nonlinear_indices
            )
            nonlinear_rows.append(
                {
                    "outcome": outcome,
                    "variable": variable,
                    "wald_chi_square": statistic,
                    "wald_df": degrees,
                    "p_value": p_value,
                }
            )
        for component in BASE_COMPONENTS[4:]:
            indices = np.asarray(
                [
                    offset + parameter_index(names, component, int(distance))
                    for distance in NODES_M[:FREE_NODE_N]
                ]
            )
            statistic, degrees, p_value = wald(beta, covariance, indices)
            rows.append(
                {
                    "outcome": outcome,
                    "component": component,
                    "component_type": "pairwise",
                    "wald_chi_square": statistic,
                    "wald_df": degrees,
                    "p_value": p_value,
                }
            )
    tests = pd.DataFrame(rows)
    nonlinear = pd.DataFrame(nonlinear_rows)
    for _, index in tests.groupby("outcome", sort=False).groups.items():
        tests.loc[index, "holm_adjusted_p"] = holm_adjust(
            tests.loc[index, "p_value"].to_numpy()
        )
    tests["passes_holm_0_05"] = tests["holm_adjusted_p"] < 0.05
    for _, index in nonlinear.groupby("outcome", sort=False).groups.items():
        nonlinear.loc[index, "holm_adjusted_p"] = holm_adjust(
            nonlinear.loc[index, "p_value"].to_numpy()
        )
    nonlinear["nonlinearity_detected_holm_0_05"] = (
        nonlinear["holm_adjusted_p"] < 0.05
    )
    previous = pd.read_csv(RESULTS / "component_whole_curve_tests_adjusted.csv")[
        ["outcome", "component", "passes_holm_0_05"]
    ].rename(columns={"passes_holm_0_05": "retained_in_linear_model"})
    tests = tests.merge(previous, on=["outcome", "component"], validate="one_to_one")
    tests["retained_after_nonlinear_main_effects"] = tests["passes_holm_0_05"]
    tests["retention_changed"] = (
        tests["retained_in_linear_model"]
        != tests["retained_after_nonlinear_main_effects"]
    )
    return tests, nonlinear


def state_membership_and_means(
    raw_climate: np.ndarray,
    components: np.ndarray,
    grid_year_weight: np.ndarray,
    pair: str,
    state: str,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = PAIR_VARIABLES[pair]
    variable_index = {variable: index for index, variable in enumerate(VARIABLES)}
    left_high = state.startswith(f"high_{left}__")
    right_high = state.endswith(f"high_{right}")
    left_value = raw_climate[:, :, variable_index[left]]
    right_value = raw_climate[:, :, variable_index[right]]
    keep = (left_value > 0 if left_high else left_value < 0) & (
        right_value > 0 if right_high else right_value < 0
    )
    flat_keep = keep.reshape(-1) & (grid_year_weight > 0)
    state_mean = np.average(
        components.reshape(-1, components.shape[2])[flat_keep],
        axis=0,
        weights=grid_year_weight[flat_keep],
    )
    return flat_keep, state_mean


def validate_state_support(
    raw_climate: np.ndarray,
    base_components: np.ndarray,
    grid_year_weight: np.ndarray,
) -> float:
    support = pd.read_csv(LEGACY_STATES / "quadrant_state_support.csv")
    maximum_difference = 0.0
    for record in support.to_dict("records"):
        keep, state_mean = state_membership_and_means(
            raw_climate,
            base_components,
            grid_year_weight,
            record["pair"],
            record["state"],
        )
        maximum_difference = max(
            maximum_difference,
            abs(int(keep.sum()) - int(record["grid_year_n"])),
            abs(
                grid_year_weight[keep].sum() / grid_year_weight.sum()
                - float(record["formal_weight_share"])
            ),
        )
        for component_index, component in enumerate(BASE_COMPONENTS):
            maximum_difference = max(
                maximum_difference,
                abs(
                    state_mean[component_index]
                    - float(record[f"component_mean__{component}"])
                ),
            )
    if maximum_difference > 1e-10:
        raise RuntimeError(
            f"quadrant state support was not reproduced: {maximum_difference}"
        )
    return maximum_difference


def state_curve(
    beta: np.ndarray,
    covariance: np.ndarray,
    names: list[str],
    components: tuple[str, ...],
    state_mean: np.ndarray,
    outcome_index: int,
) -> tuple[np.ndarray, float, int, float]:
    parameter_n = len(names)
    contrast = np.zeros((FREE_NODE_N, len(beta)))
    offset = outcome_index * parameter_n
    for node_index, distance in enumerate(NODES_M[:FREE_NODE_N]):
        for component_index, component in enumerate(components):
            contrast[
                node_index,
                offset + parameter_index(names, component, int(distance)),
            ] = state_mean[component_index]
    raw_nodes = np.r_[contrast @ beta, 0.0]
    dense_basis = node_basis(np.log10(DENSE_DISTANCE_M), np.log10(NODES_M))
    dense_curve = dense_basis @ raw_nodes
    dense_curve -= dense_curve.mean()
    estimate = contrast @ beta
    contrast_covariance = contrast @ covariance @ contrast.T
    degrees = int(np.linalg.matrix_rank(contrast_covariance))
    statistic = float(
        estimate @ np.linalg.pinv(contrast_covariance) @ estimate
    )
    return dense_curve, statistic, degrees, float(chi2.sf(statistic, degrees))


def candidate_state_comparison(
    raw_climate: np.ndarray,
    base_components: np.ndarray,
    extended_components: np.ndarray,
    grid_year_weight: np.ndarray,
    benchmark_beta: np.ndarray,
    benchmark_covariance: np.ndarray,
    extended_beta: np.ndarray,
    extended_covariance: np.ndarray,
    benchmark_names: list[str],
    extended_names: list[str],
) -> tuple[pd.DataFrame, float, float]:
    current_curves = pd.read_csv(
        LEGACY_STATES / "quadrant_state_distribution_curves.csv"
    )
    current_tests = pd.read_csv(LEGACY_STATES / "state_whole_curve_tests.csv")
    support = pd.read_csv(LEGACY_STATES / "quadrant_state_support.csv")
    rows = []
    maximum_curve_difference = 0.0
    maximum_test_difference = 0.0
    for pair, state, primary_outcome, expected_direction in CANDIDATE_STATES:
        _, base_mean = state_membership_and_means(
            raw_climate, base_components, grid_year_weight, pair, state
        )
        keep, extended_mean = state_membership_and_means(
            raw_climate, extended_components, grid_year_weight, pair, state
        )
        label = support.loc[
            support["pair"].eq(pair) & support["state"].eq(state),
            "state_label_zh",
        ].iloc[0]
        outcome_index = OUTCOMES.index(primary_outcome)
        benchmark_curve, benchmark_wald, benchmark_df, benchmark_p = state_curve(
            benchmark_beta,
            benchmark_covariance,
            benchmark_names,
            BASE_COMPONENTS,
            base_mean,
            outcome_index,
        )
        nonlinear_curve, nonlinear_wald, nonlinear_df, nonlinear_p = state_curve(
            extended_beta,
            extended_covariance,
            extended_names,
            EXTENDED_COMPONENTS,
            extended_mean,
            outcome_index,
        )
        current = current_curves.loc[
            current_curves["pair"].eq(pair)
            & current_curves["state"].eq(state)
        ].sort_values("distance_m")
        current_effect = current[
            f"{primary_outcome}_range_centered_environment_effect"
        ].to_numpy()
        maximum_curve_difference = max(
            maximum_curve_difference,
            float(np.max(np.abs(benchmark_curve - current_effect))),
        )
        current_test = current_tests.loc[
            current_tests["pair"].eq(pair)
            & current_tests["state"].eq(state)
            & current_tests["outcome"].eq(primary_outcome)
        ].iloc[0]
        maximum_test_difference = max(
            maximum_test_difference,
            abs(benchmark_wald - float(current_test["wald_chi_square"])),
            abs(benchmark_p - float(current_test["p_value"])),
        )
        correlation = float(np.corrcoef(benchmark_curve, nonlinear_curve)[0, 1])
        rows.append(
            {
                "pair": pair,
                "state": state,
                "state_label_zh": label,
                "outcome": primary_outcome,
                "outcome_label_zh": OUTCOME_LABELS[primary_outcome],
                "formal_grid_year_n": int(keep.sum()),
                "linear_effect_at_30m": benchmark_curve[0],
                "nonlinear_effect_at_30m": nonlinear_curve[0],
                "linear_whole_curve_wald_chi_square": benchmark_wald,
                "linear_whole_curve_wald_df": benchmark_df,
                "linear_whole_curve_p_value": benchmark_p,
                "nonlinear_whole_curve_wald_chi_square": nonlinear_wald,
                "nonlinear_whole_curve_wald_df": nonlinear_df,
                "nonlinear_whole_curve_p_value": nonlinear_p,
                "linear_nonlinear_curve_correlation": correlation,
                "same_30m_direction": bool(
                    np.sign(benchmark_curve[0]) == np.sign(nonlinear_curve[0])
                ),
                "expected_30m_direction_retained": bool(
                    np.sign(nonlinear_curve[0]) == expected_direction
                ),
                "primary_fingerprint_retained": bool(
                    np.sign(nonlinear_curve[0]) == expected_direction
                    and nonlinear_p < 0.05
                ),
            }
        )
    if maximum_curve_difference > 1e-9 or maximum_test_difference > 1e-8:
        raise RuntimeError(
            "benchmark model did not reproduce frozen candidate state curves"
        )
    return pd.DataFrame(rows), maximum_curve_difference, maximum_test_difference


def render_report(
    component_table: pd.DataFrame,
    nonlinear_table: pd.DataFrame,
    state_table: pd.DataFrame,
    benchmark: dict[str, float],
) -> str:
    retained_rows = []
    for outcome in OUTCOMES:
        retained = component_table.loc[
            component_table["outcome"].eq(outcome)
            & component_table["retained_after_nonlinear_main_effects"]
        ]
        retained_rows.append(
            f"| {OUTCOME_LABELS[outcome]} | "
            + "；".join(COMPONENT_LABELS[value] for value in retained["component"])
            + " |"
        )
    nonlinear_rows = "\n".join(
        f"| {OUTCOME_LABELS[row.outcome]} | {COMPONENT_LABELS[row.variable]} | "
        f"{row.wald_chi_square:.3f} | {row.wald_df} | {row.holm_adjusted_p:.3e} |"
        for row in nonlinear_table.loc[
            nonlinear_table["nonlinearity_detected_holm_0_05"]
        ].itertuples(index=False)
    )
    if not nonlinear_rows:
        nonlinear_rows = "| 无 | 无 | — | — | — |"
    state_rows = "\n".join(
        f"| {row.state_label_zh} | {row.outcome_label_zh} | "
        f"{row.linear_effect_at_30m:.3e} | {row.nonlinear_effect_at_30m:.3e} | "
        f"{row.nonlinear_whole_curve_p_value:.3e} | {row.linear_nonlinear_curve_correlation:.3f} | "
        f"{'保留' if row.primary_fingerprint_retained else '改变'} |"
        for row in state_table.itertuples(index=False)
    )
    changed = component_table.loc[component_table["retention_changed"]]
    changed_text = (
        "；".join(
            f"{OUTCOME_LABELS[row.outcome]}中的{COMPONENT_LABELS[row.component]}"
            for row in changed.itertuples(index=False)
        )
        if len(changed)
        else "无"
    )
    return f"""# 单变量非线性有效性检查

## 线性模型复现

重建模型精确恢复现有正式样本、响应方向和联合模型。线性系数最大绝对差为 `{benchmark['maximum_coefficient_difference']:.3e}`，聚类稳健协方差相对Frobenius差为 `{benchmark['covariance_relative_frobenius_difference']:.3e}`。

## 加入非线性主效应后的分量筛选

四个气候变量各使用4个响应无关分位点构造3自由度受限三次样条主效应，同时保留原有六个线性交互。每个响应内仍对十个过程分量采用Holm校正。

| 响应成分 | 校正后保留分量 |
|---|---|
{chr(10).join(retained_rows)}

与线性模型相比，候选保留状态发生变化的分量：**{changed_text}**。

## 检出的单变量非线性

| 响应成分 | 变量 | Wald χ² | 自由度 | Holm调整P值 |
|---|---|---:|---:|---:|
{nonlinear_rows}

## 原有两个候选状态

| 候选状态 | 响应成分 | 原线性30 m效应 | 非线性模型30 m效应 | 非线性整曲线P值 | 曲线相关 | 指纹 |
|---|---|---:|---:|---:|---:|---|
{state_rows}

## 决策

本检查只回答交互和候选状态是否可能由未建模的单变量弯曲关系造成。最终气候状态仍需在扩展模型的完整四维观测分布上系统投影，不能根据单个交互项直接命名机制。
"""


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUTPUT}")
    sample = pd.read_parquet(INPUTS / "formal_sample.parquet")
    climate = pd.read_parquet(INPUTS / "grid_year_climate_with_anomalies.parquet")
    arrays_all = build_long_arrays(sample)
    residual_all, mean_diagnostics = mean_residual(arrays_all, len(sample))
    expected_within = 0.0001264256879490238
    if abs(mean_diagnostics["within_effect_per_doubling"] - expected_within) > 1e-12:
        raise RuntimeError("frozen mean residual was not reproduced")

    range_pixel = sample["d_between_m"].between(30, 480).to_numpy()
    range_map = np.full(len(sample), -1, dtype=np.int32)
    range_map[range_pixel] = np.arange(range_pixel.sum(), dtype=np.int32)
    keep_observation = range_pixel[arrays_all["pixel"]]
    observation_n = int(keep_observation.sum())
    pixel_n = int(range_pixel.sum())
    grid_n = len(arrays_all["grid_levels"])
    if (observation_n, pixel_n, grid_n) != (3_557_392, 295_381, 514):
        raise RuntimeError("formal Stage A support changed")
    arrays = {
        "weight": arrays_all["weight"][keep_observation],
        "scene": arrays_all["scene"][keep_observation],
        "year": arrays_all["year"][keep_observation],
        "grid": arrays_all["grid"][keep_observation],
        "pixel": range_map[arrays_all["pixel"][keep_observation]],
        "residual": residual_all[keep_observation],
        "grid_levels": arrays_all["grid_levels"],
    }
    pixel_basis = node_basis(
        arrays_all["pixel_between_x"][range_pixel], np.log10(NODES_M)
    )
    raw_climate, climate_components, definitions, knot_registry = align_climate(
        climate, arrays["grid_levels"]
    )
    base_components = climate_components[:, :, : len(BASE_COMPONENTS)]
    grid_year = arrays["grid"] * len(YEARS) + arrays["year"]
    grid_year_weight = group_sum(
        arrays["weight"], grid_year, grid_n * len(YEARS)
    )
    support_difference = validate_state_support(
        raw_climate, base_components, grid_year_weight
    )

    positive = arrays["residual"] > 0
    negative = arrays["residual"] < 0
    if (int(positive.sum()), int(negative.sum())) != (1_889_179, 1_668_213):
        raise RuntimeError("formal residual sign counts changed")
    outcome_raw = {
        "positive_probability": positive.astype(np.float64),
        "positive_conditional_second_moment": np.square(arrays["residual"]),
        "negative_conditional_second_moment": np.square(arrays["residual"]),
    }
    outcome_indexes = {
        "positive_probability": np.arange(observation_n),
        "positive_conditional_second_moment": np.flatnonzero(positive),
        "negative_conditional_second_moment": np.flatnonzero(negative),
    }

    temporary_root = Path(tempfile.mkdtemp(prefix="nonlinear_main_check_"))
    benchmark_fits = []
    extended_fits = []
    diagnostic_rows = []
    try:
        for outcome in OUTCOMES:
            print(f"Building and fitting {outcome}", flush=True)
            benchmark_fit, extended_fit, diagnostics = fit_outcome(
                outcome,
                outcome_indexes[outcome],
                outcome_raw[outcome],
                arrays,
                pixel_basis,
                climate_components,
                temporary_root,
                pixel_n,
                grid_n * len(YEARS),
            )
            benchmark_fits.append(benchmark_fit)
            extended_fits.append(extended_fit)
            diagnostic_rows.append({"outcome": outcome, **diagnostics})
        benchmark_beta, benchmark_covariance, benchmark_correction = assemble_joint(
            benchmark_fits, observation_n, grid_n
        )
        extended_beta, extended_covariance, extended_correction = assemble_joint(
            extended_fits, observation_n, grid_n
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    frozen_beta = np.load(RESULTS / "joint_coefficient_vector.npy")
    frozen_covariance = np.load(RESULTS / "joint_cluster_robust_covariance.npy")
    maximum_coefficient_difference = float(
        np.max(np.abs(benchmark_beta - frozen_beta))
    )
    covariance_relative_difference = float(
        np.linalg.norm(benchmark_covariance - frozen_covariance)
        / np.linalg.norm(frozen_covariance)
    )
    covariance_maximum_difference = float(
        np.max(np.abs(benchmark_covariance - frozen_covariance))
    )
    if maximum_coefficient_difference > 1e-9:
        raise RuntimeError(
            f"benchmark coefficients were not reproduced: {maximum_coefficient_difference}"
        )
    if covariance_relative_difference > 1e-8:
        raise RuntimeError(
            f"benchmark covariance was not reproduced: {covariance_relative_difference}"
        )

    benchmark_names = parameter_names(BASE_COMPONENTS)
    frozen_names = pd.read_csv(RESULTS / "parameter_registry.csv")[
        "parameter"
    ].tolist()
    if benchmark_names != frozen_names:
        raise RuntimeError("benchmark parameter registry changed")
    extended_names = parameter_names(EXTENDED_COMPONENTS)
    component_table, nonlinear_table = component_tests(
        extended_beta, extended_covariance, extended_names
    )
    state_table, state_curve_difference, state_test_difference = (
        candidate_state_comparison(
            raw_climate,
            base_components,
            climate_components,
            grid_year_weight,
            benchmark_beta,
            benchmark_covariance,
            extended_beta,
            extended_covariance,
            benchmark_names,
            extended_names,
        )
    )

    benchmark_metadata = {
        "maximum_coefficient_difference": maximum_coefficient_difference,
        "covariance_relative_frobenius_difference": covariance_relative_difference,
        "maximum_covariance_element_difference": covariance_maximum_difference,
        "maximum_state_curve_difference": state_curve_difference,
        "maximum_state_test_difference": state_test_difference,
    }
    temporary_output = Path(
        tempfile.mkdtemp(prefix="nonlinear_output_", dir=RESULTS)
    )
    completed = False
    try:
        np.save(temporary_output / "joint_coefficient_vector.npy", extended_beta)
        np.save(
            temporary_output / "joint_cluster_robust_covariance.npy",
            extended_covariance,
        )
        pd.DataFrame(
            {
                "parameter_index_within_outcome": np.arange(len(extended_names)),
                "parameter": extended_names,
            }
        ).to_csv(temporary_output / "parameter_registry.csv", index=False)
        definitions.to_csv(
            temporary_output / "climate_component_definitions.csv", index=False
        )
        pd.DataFrame(diagnostic_rows).to_csv(
            temporary_output / "model_diagnostics.csv", index=False
        )
        component_table.to_csv(
            temporary_output / "component_tests_after_nonlinear_main_effects.csv",
            index=False,
        )
        nonlinear_table.to_csv(
            temporary_output / "nonlinearity_block_tests.csv", index=False
        )
        state_table.to_csv(
            temporary_output / "candidate_state_curve_comparison.csv", index=False
        )
        metadata = {
            "status": "COMPLETE_NONLINEAR_MAIN_EFFECT_CHECK",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "four flexible restricted-cubic-spline main effects plus six original linear pairwise interactions",
            "spline_knots": {
                variable: [float(value) for value in knots]
                for variable, knots in knot_registry.items()
            },
            "spline_knot_quantiles": [0.05, 0.35, 0.65, 0.95],
            "spline_basis": "linear term plus two within-grid-centered orthogonalized restricted cubic nonlinear terms",
            "observation_n": observation_n,
            "pixel_n": pixel_n,
            "grid_n": grid_n,
            "benchmark_finite_sample_correction": benchmark_correction,
            "extended_finite_sample_correction": extended_correction,
            "benchmark_reproduction": benchmark_metadata,
            "state_support_maximum_difference": support_difference,
            "all_primary_candidate_fingerprints_retained": bool(
                state_table["primary_fingerprint_retained"].all()
            ),
        }
        (temporary_output / "analysis_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )
        temporary_output.rename(OUTPUT)
        completed = True
    finally:
        if not completed:
            shutil.rmtree(temporary_output, ignore_errors=True)

    print(component_table.to_string(index=False), flush=True)
    print(state_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
