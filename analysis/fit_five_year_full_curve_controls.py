#!/usr/bin/env python3
"""Test five-year stability-curve contraction with grid and patch-window controls."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag, helmert
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_INPUT = (
    ROOT / "analysis_inputs" / "annual_climate_state" / "formal_sample.parquet"
)
PATCH_INPUT = (
    ROOT / "analysis_inputs" / "patch_context" / "focal_patch_identity.parquet"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "core_results"
    / "08_regional_stability_trend"
    / "five_year_full_curve_controls"
)

YEARS = np.arange(2013, 2026, dtype=np.int32)
WINDOW_YEARS = 5
WINDOW_STARTS = np.arange(0, len(YEARS) - WINDOW_YEARS + 1, dtype=np.int32)
WINDOW_N = len(WINDOW_STARTS)
FREE_WINDOW_N = WINDOW_N - 1
WINDOW_CENTERS = YEARS[WINDOW_STARTS + WINDOW_YEARS // 2]
DISTANCE_NODES_M = np.asarray([30.0, 60.0, 120.0, 240.0, 480.0])
DENSE_DISTANCE_M = np.geomspace(30.0, 480.0, 97)
DISTANCE_CONTRAST = helmert(len(DISTANCE_NODES_M), full=False).T
DISTANCE_DF = DISTANCE_CONTRAST.shape[1]
DRAW_N = 2000
DRAW_SEED = 20260823


@dataclass
class FitComponent:
    name: str
    beta: np.ndarray
    bread: np.ndarray
    score: np.ndarray
    parameter_n: int
    observation_n: int
    rank: int
    condition_number: float
    absorbed_weighted_r2: float
    maximum_fixed_effect_mean: float


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


def group_residualize(
    value: np.ndarray,
    weight: np.ndarray,
    group: np.ndarray,
    group_n: int,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(value, dtype=np.float64)
    original_dimension = matrix.ndim
    if original_dimension == 1:
        matrix = matrix[:, None]
    denominator = group_sum(weight, group, group_n)
    residual = matrix.copy()
    for column in range(matrix.shape[1]):
        numerator = group_sum(weight * matrix[:, column], group, group_n)
        mean = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        residual[:, column] -= mean[group]
    maximum_mean = 0.0
    for column in range(matrix.shape[1]):
        numerator = group_sum(weight * residual[:, column], group, group_n)
        mean = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        maximum_mean = max(maximum_mean, float(np.max(np.abs(mean))))
    if original_dimension == 1:
        return residual[:, 0], maximum_mean
    return residual, maximum_mean


def fit_component(
    name: str,
    design: np.ndarray,
    outcomes: np.ndarray,
    weight: np.ndarray,
    grid: np.ndarray,
    grid_n: int,
    maximum_fixed_effect_mean: float = 0.0,
) -> FitComponent:
    cross = design.T @ (design * weight[:, None])
    rank = int(np.linalg.matrix_rank(cross))
    if rank != design.shape[1]:
        raise RuntimeError(f"{name} design rank changed: {rank}/{design.shape[1]}")
    inverse = np.linalg.inv(cross)
    beta = inverse @ (design.T @ (outcomes * weight[:, None]))
    residual = outcomes - design @ beta
    parameter_n = design.shape[1]
    score = np.zeros((grid_n, parameter_n * outcomes.shape[1]), dtype=np.float64)
    for outcome in range(outcomes.shape[1]):
        for parameter in range(parameter_n):
            score[:, outcome * parameter_n + parameter] = group_sum(
                weight * design[:, parameter] * residual[:, outcome],
                grid,
                grid_n,
            )
    bread = block_diag(*([inverse] * outcomes.shape[1]))
    centered_outcomes = outcomes - np.average(outcomes, axis=0, weights=weight)
    yty = float(np.sum(weight[:, None] * np.square(centered_outcomes)))
    sse = float(np.sum(weight[:, None] * np.square(residual)))
    return FitComponent(
        name=name,
        beta=beta,
        bread=bread,
        score=score,
        parameter_n=parameter_n,
        observation_n=len(weight),
        rank=rank,
        condition_number=float(np.linalg.cond(cross)),
        absorbed_weighted_r2=float(1.0 - sse / yty) if yty > 0.0 else np.nan,
        maximum_fixed_effect_mean=maximum_fixed_effect_mean,
    )


def component_vector(component: FitComponent) -> np.ndarray:
    return np.concatenate(
        [component.beta[:, outcome] for outcome in range(component.beta.shape[1])]
    )


def stack_components(
    components: list[FitComponent], grid_n: int
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    theta = np.concatenate([component_vector(component) for component in components])
    bread = block_diag(*[component.bread for component in components])
    score = np.concatenate([component.score for component in components], axis=1)
    correction = grid_n / (grid_n - 1.0)
    covariance = correction * bread @ (score.T @ score) @ bread
    covariance = 0.5 * (covariance + covariance.T)
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for component in components:
        width = component.parameter_n * component.beta.shape[1]
        layout[component.name] = (offset, component.parameter_n)
        offset += width
    return theta, covariance, layout


def extract_component(
    theta: np.ndarray,
    layout: dict[str, tuple[int, int]],
    name: str,
) -> np.ndarray:
    offset, parameter_n = layout[name]
    return np.column_stack(
        [
            theta[offset + outcome * parameter_n : offset + (outcome + 1) * parameter_n]
            for outcome in range(2)
        ]
    )


def sum_code() -> np.ndarray:
    code = np.zeros((WINDOW_N, FREE_WINDOW_N), dtype=np.float64)
    code[:FREE_WINDOW_N] = np.eye(FREE_WINDOW_N)
    code[-1] = -1.0
    return code


def prepare_window_outcomes(
    nirv: np.ndarray,
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outcomes = np.empty((len(nirv), WINDOW_N, 2), dtype=np.float64)
    controls = np.empty((len(nirv), WINDOW_N, 2), dtype=np.float64)
    for window_index, start in enumerate(WINDOW_STARTS):
        stop = start + WINDOW_YEARS
        current = nirv[:, start:stop]
        current_scene = scene[:, start:stop]
        outcomes[:, window_index, 0] = current.mean(axis=1)
        outcomes[:, window_index, 1] = current.var(axis=1, ddof=1)
        controls[:, window_index, 0] = (current_scene == 1.0).mean(axis=1)
        controls[:, window_index, 1] = (
            (current_scene >= 2.0) & (current_scene <= 3.0)
        ).mean(axis=1)
    if np.any(outcomes[:, :, 1] <= 0.0):
        raise RuntimeError("five-year variance left positive support")
    return outcomes, controls


def fit_anchor_components(
    prefix: str,
    outcomes: np.ndarray,
    distance_features: np.ndarray,
    weight: np.ndarray,
    grid: np.ndarray,
    grid_n: int,
    fixed_effect_group: np.ndarray | None,
    fixed_effect_n: int | None,
) -> tuple[list[FitComponent], np.ndarray]:
    center_outcome = outcomes.mean(axis=1)
    level_design = np.ones((len(weight), 1), dtype=np.float64)
    level = fit_component(
        f"{prefix}_anchor_level",
        level_design,
        center_outcome,
        weight,
        grid,
        grid_n,
    )
    feature_center = np.average(distance_features, axis=0, weights=weight)
    if fixed_effect_group is None:
        group = np.zeros(len(weight), dtype=np.int32)
        group_n = 1
    else:
        group = fixed_effect_group
        if fixed_effect_n is None:
            raise RuntimeError("fixed effect count is missing")
        group_n = fixed_effect_n
    shape_design, design_error = group_residualize(
        distance_features, weight, group, group_n
    )
    shape_outcome, outcome_error = group_residualize(
        center_outcome, weight, group, group_n
    )
    shape = fit_component(
        f"{prefix}_anchor_shape",
        shape_design,
        shape_outcome,
        weight,
        grid,
        grid_n,
        max(design_error, outcome_error),
    )
    return [level, shape], feature_center


def temporal_design(
    distance_features: np.ndarray,
    controls: np.ndarray,
    include_window_main: bool,
) -> np.ndarray:
    code = sum_code()
    interaction = (
        code[None, :, :, None] * distance_features[:, None, None, :]
    ).reshape(len(distance_features), WINDOW_N, FREE_WINDOW_N * DISTANCE_DF)
    blocks = []
    if include_window_main:
        blocks.append(np.broadcast_to(code, (len(distance_features),) + code.shape))
    blocks.extend((interaction, controls - controls.mean(axis=1, keepdims=True)))
    design = np.concatenate(blocks, axis=2)
    if float(np.max(np.abs(design.mean(axis=1)))) > 1e-12:
        raise RuntimeError("temporal design lost pixel balance")
    return design


def fit_temporal_component(
    name: str,
    outcomes: np.ndarray,
    distance_features: np.ndarray,
    controls: np.ndarray,
    weight_pixel: np.ndarray,
    grid_pixel: np.ndarray,
    grid_n: int,
    group_pixel: np.ndarray | None,
    group_n: int | None,
    include_window_main: bool,
) -> FitComponent:
    design = temporal_design(distance_features, controls, include_window_main)
    outcome = outcomes - outcomes.mean(axis=1, keepdims=True)
    design = design.reshape(-1, design.shape[2])
    outcome = outcome.reshape(-1, 2)
    weight = np.repeat(weight_pixel, WINDOW_N)
    grid = np.repeat(grid_pixel, WINDOW_N)
    maximum_error = float(
        max(np.max(np.abs(outcome.reshape(-1, WINDOW_N, 2).mean(axis=1))), 0.0)
    )
    if group_pixel is not None:
        if group_n is None:
            raise RuntimeError("group count is missing")
        group_window = (
            np.repeat(group_pixel, WINDOW_N) * WINDOW_N
            + np.tile(np.arange(WINDOW_N, dtype=np.int32), len(group_pixel))
        )
        design, design_error = group_residualize(
            design, weight, group_window, group_n * WINDOW_N
        )
        outcome, outcome_error = group_residualize(
            outcome, weight, group_window, group_n * WINDOW_N
        )
        maximum_error = max(maximum_error, design_error, outcome_error)
        pixel_design_mean = design.reshape(-1, WINDOW_N, design.shape[1]).mean(axis=1)
        pixel_outcome_mean = outcome.reshape(-1, WINDOW_N, 2).mean(axis=1)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(pixel_design_mean))),
            float(np.max(np.abs(pixel_outcome_mean))),
        )
    component = fit_component(
        name,
        design,
        outcome,
        weight,
        grid,
        grid_n,
        maximum_error,
    )
    del design, outcome
    return component


def parameter_draws(
    theta: np.ndarray, covariance: np.ndarray, draw_n: int, seed: int
) -> np.ndarray:
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(eigenvalue)), 1.0) * 1e-12
    if float(np.min(eigenvalue)) < -tolerance:
        raise RuntimeError(
            f"joint covariance is not positive semidefinite: {float(np.min(eigenvalue))}"
        )
    eigenvalue = np.clip(eigenvalue, 0.0, None)
    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((draw_n, len(theta)))
    return theta + standard @ (eigenvector * np.sqrt(eigenvalue)).T


def derive_curves(
    theta: np.ndarray,
    layout: dict[str, tuple[int, int]],
    anchor_prefix: str,
    main_component: str,
    shape_component: str,
    shape_component_has_main: bool,
    dense_features: np.ndarray,
    feature_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level = extract_component(theta, layout, f"{anchor_prefix}_anchor_level")[0]
    anchor_shape = extract_component(
        theta, layout, f"{anchor_prefix}_anchor_shape"
    )
    main_beta = extract_component(theta, layout, main_component)
    main_free = main_beta[:FREE_WINDOW_N]
    main_effect = sum_code() @ main_free
    shape_beta = extract_component(theta, layout, shape_component)
    shape_start = FREE_WINDOW_N if shape_component_has_main else 0
    shape_free = shape_beta[
        shape_start : shape_start + FREE_WINDOW_N * DISTANCE_DF
    ].reshape(FREE_WINDOW_N, DISTANCE_DF, 2)
    shape_effect = np.einsum("wf,fko->wko", sum_code(), shape_free)
    centered_dense = dense_features - feature_center[None, :]
    static_curve = level[None, :] + centered_dense @ anchor_shape
    curve = (
        static_curve[None, :, :]
        + main_effect[:, None, :]
        + np.einsum("dk,wko->wdo", dense_features, shape_effect)
    )
    mean_curve = curve[:, :, 0]
    variance_curve = curve[:, :, 1]
    if np.any(mean_curve <= 0.0) or np.any(variance_curve <= 0.0):
        raise RuntimeError("derived curve left positive support")
    log_stability = np.log(mean_curve) - 0.5 * np.log(variance_curve)
    centered_log_stability = log_stability - log_stability.mean(axis=1, keepdims=True)
    strength = np.sqrt(np.mean(np.square(centered_log_stability), axis=1))
    return log_stability, centered_log_stability, strength


def valid_curve_draws(
    draws: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    strength = []
    centered_curve = []
    for draw in draws:
        try:
            _, centered, current_strength = derive_curves(draw, **kwargs)
        except RuntimeError:
            continue
        strength.append(current_strength)
        centered_curve.append(centered)
    if len(strength) < int(0.95 * len(draws)):
        raise RuntimeError(
            f"too many invalid curve draws: {len(strength)}/{len(draws)}"
        )
    return np.asarray(strength), np.asarray(centered_curve)


def coefficient_contrast(
    theta_n: int,
    layout: dict[str, tuple[int, int]],
    component_name: str,
    component_has_main: bool,
    adjacent_index: int | None,
) -> np.ndarray:
    offset, parameter_n = layout[component_name]
    shape_start = FREE_WINDOW_N if component_has_main else 0
    if adjacent_index is None:
        contrast = np.zeros(
            (2 * FREE_WINDOW_N * DISTANCE_DF, theta_n), dtype=np.float64
        )
        row = 0
        for outcome in range(2):
            outcome_offset = offset + outcome * parameter_n
            for free_window in range(FREE_WINDOW_N):
                for distance_column in range(DISTANCE_DF):
                    parameter = (
                        outcome_offset
                        + shape_start
                        + free_window * DISTANCE_DF
                        + distance_column
                    )
                    contrast[row, parameter] = 1.0
                    row += 1
        return contrast
    code = sum_code()
    difference = code[adjacent_index + 1] - code[adjacent_index]
    contrast = np.zeros((2 * DISTANCE_DF, theta_n), dtype=np.float64)
    row = 0
    for outcome in range(2):
        outcome_offset = offset + outcome * parameter_n
        for distance_column in range(DISTANCE_DF):
            for free_window, value in enumerate(difference):
                parameter = (
                    outcome_offset
                    + shape_start
                    + free_window * DISTANCE_DF
                    + distance_column
                )
                contrast[row, parameter] = value
            row += 1
    return contrast


def wald_test(
    theta: np.ndarray, covariance: np.ndarray, contrast: np.ndarray
) -> tuple[float, int, float]:
    estimate = contrast @ theta
    current_covariance = contrast @ covariance @ contrast.T
    degrees = int(np.linalg.matrix_rank(current_covariance))
    statistic = float(
        estimate @ np.linalg.pinv(current_covariance) @ estimate
    )
    return statistic, degrees, float(chi2.sf(statistic, degrees))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def draw_p_value(draw: np.ndarray) -> float:
    lower = (np.sum(draw <= 0.0) + 1.0) / (len(draw) + 1.0)
    upper = (np.sum(draw >= 0.0) + 1.0) / (len(draw) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def strict_patch_mask(
    sample: pd.DataFrame, patch: pd.DataFrame, formal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    if not np.array_equal(
        sample["pixel_id"].astype(str).to_numpy(),
        patch["pixel_id"].astype(str).to_numpy(),
    ):
        patch = patch.set_index(patch["pixel_id"].astype(str)).reindex(
            sample["pixel_id"].astype(str)
        ).reset_index(drop=True)
    distance = sample["d_between_m"].to_numpy(np.float64)
    band = np.full(len(sample), -1, dtype=np.int8)
    band[formal & (distance < 60.0)] = 0
    band[formal & (distance >= 60.0) & (distance < 120.0)] = 1
    band[formal & (distance >= 120.0) & (distance < 240.0)] = 2
    band[formal & (distance >= 240.0)] = 3
    roots = patch["baseline_patch_root"].to_numpy(np.int64)
    support = pd.DataFrame({"patch": roots[formal], "band": band[formal]})
    supported = support.drop_duplicates().groupby("patch")["band"].nunique()
    supported_roots = supported.loc[supported.eq(4)].index.to_numpy(np.int64)
    strict = formal & np.isin(roots, supported_roots)
    if len(supported_roots) != 821:
        raise RuntimeError(f"candidate patch count changed: {len(supported_roots)}")
    return strict, roots, len(supported_roots)


def sample_arrays(
    sample: pd.DataFrame,
    mask: np.ndarray,
    roots: np.ndarray | None,
) -> dict[str, object]:
    nirv = np.column_stack(
        [sample.loc[mask, f"nirv_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    scene = np.column_stack(
        [
            sample.loc[mask, f"valid_scene_n_{year}"].to_numpy(np.float64)
            for year in YEARS
        ]
    )
    outcomes, controls = prepare_window_outcomes(nirv, scene)
    weight = sample.loc[mask, "base_weight"].to_numpy(np.float64, copy=True)
    weight /= weight.mean()
    distance = sample.loc[mask, "d_between_m"].to_numpy(np.float64)
    basis = node_basis(np.log10(distance), np.log10(DISTANCE_NODES_M))
    features = basis @ DISTANCE_CONTRAST
    grid, grid_levels = pd.factorize(
        sample.loc[mask, "grid_id"].astype(str), sort=True
    )
    result: dict[str, object] = {
        "outcomes": outcomes,
        "controls": controls,
        "weight": weight,
        "distance": distance,
        "features": features,
        "grid": grid.astype(np.int32),
        "grid_levels": grid_levels,
        "pixel_n": int(mask.sum()),
    }
    if roots is not None:
        patch_index, patch_levels = pd.factorize(roots[mask], sort=True)
        result["patch"] = patch_index.astype(np.int32)
        result["patch_levels"] = patch_levels
    return result


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUTPUT}")
    columns = ["pixel_id", "grid_id", "base_weight", "d_between_m"]
    columns += [f"nirv_{year}" for year in YEARS]
    columns += [f"distance_m_{year}" for year in YEARS]
    columns += [f"valid_scene_n_{year}" for year in YEARS]
    sample = pd.read_parquet(SAMPLE_INPUT, columns=columns)
    patch = pd.read_parquet(PATCH_INPUT)
    nirv_all = np.column_stack(
        [sample[f"nirv_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    distance_all = np.column_stack(
        [sample[f"distance_m_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    scene_all = np.column_stack(
        [sample[f"valid_scene_n_{year}"].to_numpy(np.float64) for year in YEARS]
    )
    formal = sample["d_between_m"].between(30.0, 480.0).to_numpy()
    valid = (
        np.isfinite(nirv_all)
        & np.isfinite(distance_all)
        & (distance_all > 0.0)
        & (scene_all >= 1.0)
    )
    balanced = formal & valid.all(axis=1)
    if int(balanced.sum()) != 129_307:
        raise RuntimeError(f"balanced support changed: {int(balanced.sum())}")
    strict, roots, candidate_patch_n = strict_patch_mask(sample, patch, formal)
    patch_balanced = balanced & strict
    full = sample_arrays(sample, balanced, None)
    same_patch = sample_arrays(sample, patch_balanced, roots)
    if (same_patch["pixel_n"], len(same_patch["patch_levels"])) != (97_358, 678):
        raise RuntimeError(
            "complete-panel patch support changed: "
            f"{same_patch['pixel_n']}, {len(same_patch['patch_levels'])}"
        )
    dense_basis = node_basis(
        np.log10(DENSE_DISTANCE_M), np.log10(DISTANCE_NODES_M)
    )
    dense_features = dense_basis @ DISTANCE_CONTRAST

    print("fit full-sample anchors", flush=True)
    full_global_anchor, full_feature_center = fit_anchor_components(
        "full_global",
        full["outcomes"],
        full["features"],
        full["weight"],
        full["grid"],
        len(full["grid_levels"]),
        None,
        None,
    )
    full_grid_anchor, full_grid_feature_center = fit_anchor_components(
        "full_grid",
        full["outcomes"],
        full["features"],
        full["weight"],
        full["grid"],
        len(full["grid_levels"]),
        full["grid"],
        len(full["grid_levels"]),
    )
    print("fit full-sample temporal models", flush=True)
    full_baseline = fit_temporal_component(
        "full_temporal_baseline",
        full["outcomes"],
        full["features"],
        full["controls"],
        full["weight"],
        full["grid"],
        len(full["grid_levels"]),
        None,
        None,
        True,
    )
    full_grid_window = fit_temporal_component(
        "full_temporal_grid_window",
        full["outcomes"],
        full["features"],
        full["controls"],
        full["weight"],
        full["grid"],
        len(full["grid_levels"]),
        full["grid"],
        len(full["grid_levels"]),
        False,
    )

    print("fit complete-panel same-patch anchors", flush=True)
    patch_anchor, patch_feature_center = fit_anchor_components(
        "patch",
        same_patch["outcomes"],
        same_patch["features"],
        same_patch["weight"],
        same_patch["grid"],
        len(same_patch["grid_levels"]),
        same_patch["patch"],
        len(same_patch["patch_levels"]),
    )
    print("fit complete-panel same-patch temporal models", flush=True)
    patch_baseline = fit_temporal_component(
        "patch_temporal_baseline",
        same_patch["outcomes"],
        same_patch["features"],
        same_patch["controls"],
        same_patch["weight"],
        same_patch["grid"],
        len(same_patch["grid_levels"]),
        None,
        None,
        True,
    )
    patch_window = fit_temporal_component(
        "patch_temporal_patch_window",
        same_patch["outcomes"],
        same_patch["features"],
        same_patch["controls"],
        same_patch["weight"],
        same_patch["grid"],
        len(same_patch["grid_levels"]),
        same_patch["patch"],
        len(same_patch["patch_levels"]),
        False,
    )

    specifications = [
        {
            "key": "full_original",
            "label": "固定完整像元：原五年窗口曲线",
            "components": [*full_global_anchor, full_baseline],
            "grid_n": len(full["grid_levels"]),
            "anchor_prefix": "full_global",
            "main_component": "full_temporal_baseline",
            "shape_component": "full_temporal_baseline",
            "shape_component_has_main": True,
            "feature_center": full_feature_center,
            "pixel_n": full["pixel_n"],
            "group_n": len(full["grid_levels"]),
        },
        {
            "key": "full_grid_window",
            "label": "固定完整像元：网格×窗口控制",
            "components": [*full_grid_anchor, full_baseline, full_grid_window],
            "grid_n": len(full["grid_levels"]),
            "anchor_prefix": "full_grid",
            "main_component": "full_temporal_baseline",
            "shape_component": "full_temporal_grid_window",
            "shape_component_has_main": False,
            "feature_center": full_grid_feature_center,
            "pixel_n": full["pixel_n"],
            "group_n": len(full["grid_levels"]),
        },
        {
            "key": "same_patch_baseline",
            "label": "完整面板同斑块样本：同样本对照",
            "components": [*patch_anchor, patch_baseline],
            "grid_n": len(same_patch["grid_levels"]),
            "anchor_prefix": "patch",
            "main_component": "patch_temporal_baseline",
            "shape_component": "patch_temporal_baseline",
            "shape_component_has_main": True,
            "feature_center": patch_feature_center,
            "pixel_n": same_patch["pixel_n"],
            "group_n": len(same_patch["patch_levels"]),
        },
        {
            "key": "same_patch_patch_window",
            "label": "完整面板同斑块样本：斑块×窗口控制",
            "components": [*patch_anchor, patch_baseline, patch_window],
            "grid_n": len(same_patch["grid_levels"]),
            "anchor_prefix": "patch",
            "main_component": "patch_temporal_baseline",
            "shape_component": "patch_temporal_patch_window",
            "shape_component_has_main": False,
            "feature_center": patch_feature_center,
            "pixel_n": same_patch["pixel_n"],
            "group_n": len(same_patch["patch_levels"]),
        },
    ]

    curve_rows = []
    strength_rows = []
    global_rows = []
    adjacent_rows = []
    diagnostics_rows = []
    for specification_index, specification in enumerate(specifications):
        print(f"derive {specification['key']}", flush=True)
        theta, covariance, layout = stack_components(
            specification["components"], specification["grid_n"]
        )
        kwargs = {
            "layout": layout,
            "anchor_prefix": specification["anchor_prefix"],
            "main_component": specification["main_component"],
            "shape_component": specification["shape_component"],
            "shape_component_has_main": specification["shape_component_has_main"],
            "dense_features": dense_features,
            "feature_center": specification["feature_center"],
        }
        log_stability, centered_curve, strength = derive_curves(theta, **kwargs)
        draws = parameter_draws(
            theta, covariance, DRAW_N, DRAW_SEED + specification_index
        )
        strength_draws, centered_draws = valid_curve_draws(draws, **kwargs)
        relative = 100.0 * (strength / strength[0] - 1.0)
        relative_draws = 100.0 * (
            strength_draws / strength_draws[:, [0]] - 1.0
        )
        strength_lower, strength_upper = np.quantile(
            strength_draws, [0.025, 0.975], axis=0
        )
        relative_lower, relative_upper = np.quantile(
            relative_draws, [0.025, 0.975], axis=0
        )
        centered_lower, centered_upper = np.quantile(
            centered_draws, [0.025, 0.975], axis=0
        )
        for window_index, center_year in enumerate(WINDOW_CENTERS):
            strength_rows.append(
                {
                    "specification": specification["key"],
                    "specification_label_zh": specification["label"],
                    "window_start_year": int(YEARS[WINDOW_STARTS[window_index]]),
                    "window_end_year": int(
                        YEARS[WINDOW_STARTS[window_index] + WINDOW_YEARS - 1]
                    ),
                    "window_center_year": int(center_year),
                    "whole_curve_spatial_strength_log_rms": strength[window_index],
                    "strength_lower_95": strength_lower[window_index],
                    "strength_upper_95": strength_upper[window_index],
                    "relative_to_first_window_percent": relative[window_index],
                    "relative_lower_95": relative_lower[window_index],
                    "relative_upper_95": relative_upper[window_index],
                    "valid_parameter_draw_n": len(strength_draws),
                }
            )
            for distance_index, current_distance in enumerate(DENSE_DISTANCE_M):
                curve_rows.append(
                    {
                        "specification": specification["key"],
                        "specification_label_zh": specification["label"],
                        "window_start_year": int(
                            YEARS[WINDOW_STARTS[window_index]]
                        ),
                        "window_end_year": int(
                            YEARS[WINDOW_STARTS[window_index] + WINDOW_YEARS - 1]
                        ),
                        "window_center_year": int(center_year),
                        "distance_m": current_distance,
                        "temporal_stability": np.exp(
                            log_stability[window_index, distance_index]
                        ),
                        "centered_log_stability": centered_curve[
                            window_index, distance_index
                        ],
                        "relative_to_curve_geometric_mean_percent": 100.0
                        * np.expm1(centered_curve[window_index, distance_index]),
                        "centered_log_stability_lower_95": centered_lower[
                            window_index, distance_index
                        ],
                        "centered_log_stability_upper_95": centered_upper[
                            window_index, distance_index
                        ],
                    }
                )
        global_contrast = coefficient_contrast(
            len(theta),
            layout,
            specification["shape_component"],
            specification["shape_component_has_main"],
            None,
        )
        statistic, degrees, p_value = wald_test(theta, covariance, global_contrast)
        global_rows.append(
            {
                "specification": specification["key"],
                "specification_label_zh": specification["label"],
                "test": "joint_window_by_continuous_distance_for_mean_and_variance",
                "whole_curve_wald_chi_square": statistic,
                "whole_curve_wald_df": degrees,
                "whole_curve_p_value": p_value,
                "pixel_n": specification["pixel_n"],
                "group_n": specification["group_n"],
                "grid_n": specification["grid_n"],
            }
        )
        current_adjacent_rows = []
        current_p = []
        current_strength_p = []
        for adjacent_index in range(WINDOW_N - 1):
            contrast = coefficient_contrast(
                len(theta),
                layout,
                specification["shape_component"],
                specification["shape_component_has_main"],
                adjacent_index,
            )
            statistic, degrees, p_value = wald_test(theta, covariance, contrast)
            current_p.append(p_value)
            current_strength_change_draw = 100.0 * (
                strength_draws[:, adjacent_index + 1]
                / strength_draws[:, adjacent_index]
                - 1.0
            )
            strength_change_lower, strength_change_upper = np.quantile(
                current_strength_change_draw, [0.025, 0.975]
            )
            strength_change_p = draw_p_value(current_strength_change_draw)
            current_strength_p.append(strength_change_p)
            current_adjacent_rows.append(
                {
                    "specification": specification["key"],
                    "specification_label_zh": specification["label"],
                    "from_window": (
                        f"{YEARS[WINDOW_STARTS[adjacent_index]]}--"
                        f"{YEARS[WINDOW_STARTS[adjacent_index] + WINDOW_YEARS - 1]}"
                    ),
                    "to_window": (
                        f"{YEARS[WINDOW_STARTS[adjacent_index + 1]]}--"
                        f"{YEARS[WINDOW_STARTS[adjacent_index + 1] + WINDOW_YEARS - 1]}"
                    ),
                    "whole_curve_wald_chi_square": statistic,
                    "whole_curve_wald_df": degrees,
                    "whole_curve_p_value": p_value,
                    "spatial_strength_change_percent": 100.0
                    * (strength[adjacent_index + 1] / strength[adjacent_index] - 1.0),
                    "spatial_strength_change_lower_95": strength_change_lower,
                    "spatial_strength_change_upper_95": strength_change_upper,
                    "spatial_strength_change_p_value": strength_change_p,
                }
            )
        adjusted = holm_adjust(np.asarray(current_p))
        strength_adjusted = holm_adjust(np.asarray(current_strength_p))
        for row, adjusted_p, strength_adjusted_p in zip(
            current_adjacent_rows, adjusted, strength_adjusted
        ):
            row["whole_curve_holm_adjusted_p"] = adjusted_p
            row["whole_curve_retained_holm_0_05"] = bool(adjusted_p < 0.05)
            row["spatial_strength_holm_adjusted_p"] = strength_adjusted_p
            row["spatial_strength_retained_holm_0_05"] = bool(
                strength_adjusted_p < 0.05
            )
            adjacent_rows.append(row)
        for component in specification["components"]:
            diagnostics_rows.append(
                {
                    "specification": specification["key"],
                    "component": component.name,
                    "observation_n": component.observation_n,
                    "parameter_n": component.parameter_n,
                    "rank": component.rank,
                    "condition_number": component.condition_number,
                    "absorbed_weighted_r2": component.absorbed_weighted_r2,
                    "maximum_fixed_effect_mean": component.maximum_fixed_effect_mean,
                }
            )

    temporary_root = Path(tempfile.mkdtemp(prefix="five_year_full_curve_controls_"))
    temporary_output = temporary_root / OUTPUT.name
    temporary_output.mkdir()
    try:
        pd.DataFrame(curve_rows).to_csv(
            temporary_output / "full_curve_stability_by_window.csv", index=False
        )
        pd.DataFrame(strength_rows).to_csv(
            temporary_output / "whole_curve_spatial_strength_by_window.csv",
            index=False,
        )
        pd.DataFrame(global_rows).to_csv(
            temporary_output / "window_distance_global_tests.csv", index=False
        )
        pd.DataFrame(adjacent_rows).to_csv(
            temporary_output / "adjacent_window_whole_curve_tests.csv", index=False
        )
        pd.DataFrame(diagnostics_rows).drop_duplicates().to_csv(
            temporary_output / "model_diagnostics.csv", index=False
        )
        support = pd.DataFrame(
            [
                {
                    "sample": "fixed_complete_panel",
                    "pixel_n": full["pixel_n"],
                    "patch_n": np.nan,
                    "grid_n": len(full["grid_levels"]),
                },
                {
                    "sample": "candidate_all_distance_patches",
                    "pixel_n": int(strict.sum()),
                    "patch_n": candidate_patch_n,
                    "grid_n": int(sample.loc[strict, "grid_id"].nunique()),
                },
                {
                    "sample": "fixed_complete_panel_within_candidate_patches",
                    "pixel_n": same_patch["pixel_n"],
                    "patch_n": len(same_patch["patch_levels"]),
                    "grid_n": len(same_patch["grid_levels"]),
                },
            ]
        )
        support.to_csv(temporary_output / "analysis_support.csv", index=False)
        metadata = {
            "status": "COMPLETE_FIVE_YEAR_FULL_CURVE_CONTROLS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [
                str(SAMPLE_INPUT.relative_to(ROOT)),
                str(PATCH_INPUT.relative_to(ROOT)),
            ],
            "years": YEARS.tolist(),
            "rolling_window_years": WINDOW_YEARS,
            "windows": [
                [
                    int(YEARS[start]),
                    int(YEARS[start + WINDOW_YEARS - 1]),
                ]
                for start in WINDOW_STARTS
            ],
            "dense_distance_point_n": len(DENSE_DISTANCE_M),
            "dense_distance_weighting": "equal weight across 97 log-spaced points from 30 to 480 m",
            "spatial_strength_estimand": "root mean square of the 97-point log-stability curve after subtracting that window's 97-point curve mean",
            "relative_change_estimand": "percent change in whole-curve spatial strength relative to the first 2013--2017 window",
            "fixed_complete_panel_pixel_n": full["pixel_n"],
            "fixed_complete_panel_grid_n": len(full["grid_levels"]),
            "candidate_all_distance_patch_n": candidate_patch_n,
            "complete_panel_same_patch_pixel_n": same_patch["pixel_n"],
            "complete_panel_same_patch_n": len(same_patch["patch_levels"]),
            "patch_support_note": "The original 821 all-distance patches contain 214,841 formal pixels, but intersection with the fixed 13-year complete panel contains 97,358 pixels in 678 patches; fixed pixel support was prioritized.",
            "models": {
                "full_original": "pixel fixed effects, five-year window by neutral four-dimensional continuous-distance contrasts, and window-level scene-quality controls",
                "full_grid_window": "full_original time levels plus within-grid static distance anchor and grid-by-window fixed effects for distance-shape change",
                "same_patch_baseline": "same 97,358-pixel complete-panel sample and within-patch static distance anchor, without patch-by-window fixed effects",
                "same_patch_patch_window": "same_patch_baseline time levels plus patch-by-window fixed effects for distance-shape change",
            },
            "inference": "50-km-grid cluster-robust joint covariance for mean and variance; 2,000 multivariate-normal coefficient draws for nonlinear curve-strength intervals",
            "draw_seed": DRAW_SEED,
            "whole_curve_test": "joint Wald test of all window-by-neutral-distance interaction coefficients for five-year mean and variance; no endpoint or near-far contrast",
            "adjacent_tests": "eight adjacent-window joint whole-curve tests with Holm correction",
            "cloud_tasks_started": False,
            "lst_or_canopy_read": False,
            "formal_manuscript_documents_updated": False,
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
    main()
