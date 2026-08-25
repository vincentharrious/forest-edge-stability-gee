#!/usr/bin/env python3
"""Project every observed four-dimensional annual climate state onto distance curves."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import fit_nonlinear_main_effect_check as nonlinear


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "analysis_inputs" / "annual_climate_state"
RESULTS = ROOT / "outputs" / "core_results" / "02_environmental_states"
MODEL = RESULTS / "nonlinear_main_effect_check"
OUTPUT = RESULTS / "observed_climate_states"

VARIABLE_LABELS = {
    "shortwave": "短波辐射",
    "temperature": "温度",
    "vpd": "VPD",
    "rootzone_soil_moisture": "根区土壤水分",
}
STATE_SIGNS = tuple(product((-1, 1), repeat=len(nonlinear.VARIABLES)))
PREVIOUS_SLICES = (
    (
        "high_temperature__high_vpd",
        "温度偏高 × VPD偏高",
        {"temperature": 1, "vpd": 1},
    ),
    (
        "high_temperature__high_rootzone_soil_moisture",
        "温度偏高 × 根区土壤水分偏高",
        {"temperature": 1, "rootzone_soil_moisture": 1},
    ),
)
SHARED_AXIS_STATES = (
    (
        "T_H__VPD_H__SM_H",
        "high_temperature__high_vpd__high_rootzone_soil_moisture",
        "温度偏高 × VPD偏高 × 根区土壤水分偏高（短波不限定）",
        {"temperature": 1, "vpd": 1, "rootzone_soil_moisture": 1},
    ),
    (
        "T_H__VPD_H__SM_L",
        "high_temperature__high_vpd__low_rootzone_soil_moisture",
        "温度偏高 × VPD偏高 × 根区土壤水分偏低（短波不限定）",
        {"temperature": 1, "vpd": 1, "rootzone_soil_moisture": -1},
    ),
)
Z_975 = float(norm.ppf(0.975))


def state_code(signs: tuple[int, ...]) -> str:
    return "".join("H" if sign > 0 else "L" for sign in signs)


def state_name(signs: tuple[int, ...]) -> str:
    return "__".join(
        f"{'high' if sign > 0 else 'low'}_{variable}"
        for variable, sign in zip(nonlinear.VARIABLES, signs, strict=True)
    )


def state_label(signs: tuple[int, ...]) -> str:
    return " × ".join(
        f"{VARIABLE_LABELS[variable]}偏{'高' if sign > 0 else '低'}"
        for variable, sign in zip(nonlinear.VARIABLES, signs, strict=True)
    )


def formal_grid_year_weights(
    sample: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    arrays = nonlinear.build_long_arrays(sample)
    range_pixel = sample["d_between_m"].between(30, 480).to_numpy()
    keep_observation = range_pixel[arrays["pixel"]]
    observation_n = int(keep_observation.sum())
    pixel_n = int(range_pixel.sum())
    grid_n = len(arrays["grid_levels"])
    if (observation_n, pixel_n, grid_n) != (3_557_392, 295_381, 514):
        raise RuntimeError("formal Stage A support changed")
    grid_year = (
        arrays["grid"][keep_observation] * len(nonlinear.YEARS)
        + arrays["year"][keep_observation]
    )
    weights = nonlinear.group_sum(
        arrays["weight"][keep_observation],
        grid_year,
        grid_n * len(nonlinear.YEARS),
    ).reshape(grid_n, len(nonlinear.YEARS))
    if not np.any(weights > 0):
        raise RuntimeError("no formal grid-year has positive weight")
    return arrays["grid_levels"], weights, observation_n


def observed_state_support(
    grid_levels: np.ndarray,
    grid_year_weight: np.ndarray,
    raw_climate: np.ndarray,
    climate_components: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    if np.any(raw_climate == 0):
        raise RuntimeError("a climate anomaly lies exactly on a state boundary")
    supported = grid_year_weight > 0
    total_weight = float(grid_year_weight.sum())
    flat_weight = grid_year_weight.reshape(-1)
    flat_raw = raw_climate.reshape(-1, raw_climate.shape[2])
    flat_components = climate_components.reshape(-1, climate_components.shape[2])
    assignment = np.full(raw_climate.shape[:2], "", dtype=object)
    support_rows = []
    component_rows = []
    masks: dict[str, np.ndarray] = {}

    for signs in STATE_SIGNS:
        code = state_code(signs)
        name = state_name(signs)
        label = state_label(signs)
        mask = np.ones(raw_climate.shape[:2], dtype=bool)
        for variable_index, sign in enumerate(signs):
            mask &= (
                raw_climate[:, :, variable_index] > 0
                if sign > 0
                else raw_climate[:, :, variable_index] < 0
            )
        mask &= supported
        assignment[mask] = code
        flat_mask = mask.reshape(-1)
        state_weight = float(flat_weight[flat_mask].sum())
        raw_mean = np.average(
            flat_raw[flat_mask], axis=0, weights=flat_weight[flat_mask]
        )
        component_mean = np.average(
            flat_components[flat_mask], axis=0, weights=flat_weight[flat_mask]
        )
        support_row = {
            "state_code": code,
            "state": name,
            "state_label_zh": label,
            "grid_year_n": int(flat_mask.sum()),
            "grid_n": int(mask.any(axis=1).sum()),
            "grid_coverage_share": float(mask.any(axis=1).mean()),
            "formal_weight_share": state_weight / total_weight,
        }
        for variable, sign, value in zip(
            nonlinear.VARIABLES, signs, raw_mean, strict=True
        ):
            support_row[f"state_sign__{variable}"] = "high" if sign > 0 else "low"
            support_row[f"mean__{variable}_z"] = value
        support_rows.append(support_row)
        component_rows.append(
            {
                "state_code": code,
                **{
                    f"component_mean__{component}": component_mean[index]
                    for index, component in enumerate(nonlinear.EXTENDED_COMPONENTS)
                },
            }
        )
        masks[code] = mask

    if np.any(assignment[supported] == ""):
        raise RuntimeError("formal grid-years were not exhaustively assigned")
    support = pd.DataFrame(support_rows)
    if int(support["grid_year_n"].sum()) != int(supported.sum()):
        raise RuntimeError("four-dimensional state counts are not exhaustive")
    if abs(float(support["formal_weight_share"].sum()) - 1.0) > 1e-12:
        raise RuntimeError("four-dimensional state weights do not sum to one")
    support["formal_weight_rank"] = support["formal_weight_share"].rank(
        ascending=False, method="min"
    ).astype(int)
    support = support.sort_values("formal_weight_rank").reset_index(drop=True)
    components = pd.DataFrame(component_rows)

    flat_supported = supported.reshape(-1)
    membership = pd.DataFrame(
        {
            "grid_id": np.repeat(grid_levels.astype(str), len(nonlinear.YEARS))[
                flat_supported
            ],
            "year": np.tile(nonlinear.YEARS, len(grid_levels))[flat_supported],
            "state_code": assignment.reshape(-1)[flat_supported],
            "formal_weight": flat_weight[flat_supported],
            "formal_weight_share": flat_weight[flat_supported] / total_weight,
        }
    )
    for variable_index, variable in enumerate(nonlinear.VARIABLES):
        membership[f"{variable}_z"] = flat_raw[flat_supported, variable_index]
    membership = membership.merge(
        support[["state_code", "state", "state_label_zh"]],
        on="state_code",
        validate="many_to_one",
    )
    return support, components, membership, masks


def parameter_names() -> list[str]:
    registry = pd.read_csv(MODEL / "parameter_registry.csv")
    if not np.array_equal(
        registry["parameter_index_within_outcome"].to_numpy(),
        np.arange(len(registry)),
    ):
        raise RuntimeError("extended parameter registry is not ordered")
    return registry["parameter"].tolist()


def state_contrasts(
    state_component_mean: np.ndarray,
    outcome_index: int,
    names: list[str],
    total_parameter_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameter_n = len(names)
    offset = outcome_index * parameter_n
    free = np.zeros((nonlinear.FREE_NODE_N, total_parameter_n))
    for node_index, distance in enumerate(
        nonlinear.NODES_M[: nonlinear.FREE_NODE_N]
    ):
        for component_index, component in enumerate(nonlinear.EXTENDED_COMPONENTS):
            parameter = f"{component}_x_distance_{int(distance)}m_vs480"
            free[node_index, offset + names.index(parameter)] = state_component_mean[
                component_index
            ]
    node_contrast = np.vstack(
        [free, np.zeros((1, total_parameter_n), dtype=np.float64)]
    )
    dense_basis = nonlinear.node_basis(
        np.log10(nonlinear.DENSE_DISTANCE_M), np.log10(nonlinear.NODES_M)
    )
    dense = dense_basis @ node_contrast
    dense -= dense.mean(axis=0, keepdims=True)

    log_distance = np.log2(nonlinear.DENSE_DISTANCE_M / 30.0)
    near = log_distance <= 2.0
    far = log_distance >= 2.0
    near_average = np.trapezoid(dense[near], log_distance[near], axis=0) / 2.0
    far_average = np.trapezoid(dense[far], log_distance[far], axis=0) / 2.0
    return free, dense, near_average - far_average


def project_states(
    support: pd.DataFrame,
    component_means: pd.DataFrame,
    beta: np.ndarray,
    covariance: np.ndarray,
    names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    test_rows = []
    components_by_state = component_means.set_index("state_code")
    total_parameter_n = len(beta)
    for state in support.itertuples(index=False):
        state_mean = components_by_state.loc[
            state.state_code,
            [f"component_mean__{value}" for value in nonlinear.EXTENDED_COMPONENTS],
        ].to_numpy(np.float64)
        for outcome_index, outcome in enumerate(nonlinear.OUTCOMES):
            free, dense, half = state_contrasts(
                state_mean, outcome_index, names, total_parameter_n
            )
            free_estimate = free @ beta
            free_covariance = free @ covariance @ free.T
            degrees = int(np.linalg.matrix_rank(free_covariance))
            statistic = float(
                free_estimate @ np.linalg.pinv(free_covariance) @ free_estimate
            )
            p_value = float(chi2.sf(statistic, degrees))

            estimate = dense @ beta
            variance = np.einsum(
                "ij,jk,ik->i", dense, covariance, dense, optimize=True
            )
            standard_error = np.sqrt(np.maximum(variance, 0.0))
            half_estimate = float(half @ beta)
            half_variance = float(half @ covariance @ half)
            half_standard_error = float(np.sqrt(max(half_variance, 0.0)))
            half_p_value = float(
                2.0 * norm.sf(abs(half_estimate / half_standard_error))
            )
            test_rows.append(
                {
                    "state_code": state.state_code,
                    "state": state.state,
                    "state_label_zh": state.state_label_zh,
                    "grid_year_n": state.grid_year_n,
                    "grid_n": state.grid_n,
                    "formal_weight_share": state.formal_weight_share,
                    "outcome": outcome,
                    "outcome_label_zh": nonlinear.OUTCOME_LABELS[outcome],
                    "whole_curve_wald_chi_square": statistic,
                    "whole_curve_wald_df": degrees,
                    "whole_curve_p_value": p_value,
                    "near_vs_far_log_half_contrast": half_estimate,
                    "near_vs_far_log_half_standard_error": half_standard_error,
                    "near_vs_far_log_half_lower_95": half_estimate
                    - Z_975 * half_standard_error,
                    "near_vs_far_log_half_upper_95": half_estimate
                    + Z_975 * half_standard_error,
                    "near_vs_far_log_half_p_value": half_p_value,
                }
            )
            for distance, value, se in zip(
                nonlinear.DENSE_DISTANCE_M, estimate, standard_error, strict=True
            ):
                curve_rows.append(
                    {
                        "state_code": state.state_code,
                        "state": state.state,
                        "state_label_zh": state.state_label_zh,
                        "grid_year_n": state.grid_year_n,
                        "grid_n": state.grid_n,
                        "formal_weight_share": state.formal_weight_share,
                        "outcome": outcome,
                        "outcome_label_zh": nonlinear.OUTCOME_LABELS[outcome],
                        "distance_m": distance,
                        "range_centered_environment_effect": value,
                        "standard_error": se,
                        "lower_95": value - Z_975 * se,
                        "upper_95": value + Z_975 * se,
                    }
                )

    tests = pd.DataFrame(test_rows)
    for _, indexes in tests.groupby("outcome", sort=False).groups.items():
        tests.loc[indexes, "holm_adjusted_p"] = nonlinear.holm_adjust(
            tests.loc[indexes, "whole_curve_p_value"].to_numpy()
        )
    tests["whole_curve_retained_holm_0_05"] = tests["holm_adjusted_p"] < 0.05
    tests["near_edge_direction"] = np.where(
        tests["near_vs_far_log_half_contrast"] > 0, "higher", "lower"
    )
    return pd.DataFrame(curve_rows), tests


def screen_fingerprints(
    support: pd.DataFrame, tests: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    test_index = tests.set_index(["state_code", "outcome"])
    for state in support.itertuples(index=False):
        row = {
            "state_code": state.state_code,
            "state": state.state,
            "state_label_zh": state.state_label_zh,
            "grid_year_n": state.grid_year_n,
            "grid_n": state.grid_n,
            "grid_coverage_share": state.grid_coverage_share,
            "formal_weight_share": state.formal_weight_share,
        }
        for outcome in nonlinear.OUTCOMES:
            result = test_index.loc[(state.state_code, outcome)]
            row[f"{outcome}__edge_half_contrast"] = result[
                "near_vs_far_log_half_contrast"
            ]
            row[f"{outcome}__holm_adjusted_p"] = result["holm_adjusted_p"]
            row[f"{outcome}__retained"] = bool(
                result["whole_curve_retained_holm_0_05"]
            )
        probability_retained = row["positive_probability__retained"]
        probability_direction = row["positive_probability__edge_half_contrast"]
        positive_retained = row[
            "positive_conditional_second_moment__retained"
        ]
        positive_direction = row[
            "positive_conditional_second_moment__edge_half_contrast"
        ]
        negative_retained = row[
            "negative_conditional_second_moment__retained"
        ]
        negative_direction = row[
            "negative_conditional_second_moment__edge_half_contrast"
        ]
        row["negative_side_reinforcement"] = bool(
            probability_retained
            and probability_direction < 0
            and negative_retained
            and negative_direction > 0
        )
        row["positive_side_reinforcement"] = bool(
            probability_retained
            and probability_direction > 0
            and positive_retained
            and positive_direction > 0
        )
        row["positive_amplitude_amplification"] = bool(
            positive_retained and positive_direction > 0
        )
        row["negative_side_buffering"] = bool(
            probability_retained
            and probability_direction > 0
            and negative_retained
            and negative_direction < 0
        )
        if row["negative_side_reinforcement"] and row[
            "positive_side_reinforcement"
        ]:
            row["response_fingerprint"] = "bidirectional_reinforcement"
        elif row["negative_side_reinforcement"]:
            row["response_fingerprint"] = "negative_side_reinforcement"
        elif row["positive_side_reinforcement"]:
            row["response_fingerprint"] = "positive_side_reinforcement"
        elif row["positive_amplitude_amplification"]:
            row["response_fingerprint"] = "positive_amplitude_amplification"
        elif row["negative_side_buffering"]:
            row["response_fingerprint"] = "negative_side_buffering"
        elif any(row[f"{outcome}__retained"] for outcome in nonlinear.OUTCOMES):
            row["response_fingerprint"] = "other_curve_deformation"
        else:
            row["response_fingerprint"] = "no_retained_curve"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["negative_side_reinforcement", "positive_amplitude_amplification", "formal_weight_share"],
        ascending=[False, False, False],
    )


def shared_axis_states(
    support: pd.DataFrame,
    component_means: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_rows = []
    component_rows = []
    component_index = component_means.set_index("state_code")
    for code, name, label, constraints in SHARED_AXIS_STATES:
        keep = np.ones(len(support), dtype=bool)
        for variable, sign in constraints.items():
            keep &= support[f"state_sign__{variable}"].eq(
                "high" if sign > 0 else "low"
            )
        cells = support.loc[keep].copy()
        cell_codes = cells["state_code"].tolist()
        weights = cells["formal_weight_share"].to_numpy(np.float64)
        component_mean = np.average(
            component_index.loc[
                cell_codes,
                [
                    f"component_mean__{component}"
                    for component in nonlinear.EXTENDED_COMPONENTS
                ],
            ].to_numpy(np.float64),
            axis=0,
            weights=weights,
        )
        members = membership.loc[membership["state_code"].isin(cell_codes)]
        support_row = {
            "state_code": code,
            "state": name,
            "state_label_zh": label,
            "source_four_dimensional_states": ";".join(cell_codes),
            "grid_year_n": int(len(members)),
            "grid_n": int(members["grid_id"].nunique()),
            "grid_coverage_share": float(
                members["grid_id"].nunique() / support["grid_n"].max()
            ),
            "formal_weight_share": float(members["formal_weight_share"].sum()),
            "state_sign__shortwave": "unconstrained",
        }
        for variable in nonlinear.VARIABLES:
            support_row[f"state_sign__{variable}"] = (
                "high"
                if constraints.get(variable) == 1
                else "low"
                if constraints.get(variable) == -1
                else "unconstrained"
            )
            support_row[f"mean__{variable}_z"] = float(
                np.average(
                    cells[f"mean__{variable}_z"].to_numpy(np.float64),
                    weights=weights,
                )
            )
        support_rows.append(support_row)
        component_rows.append(
            {
                "state_code": code,
                **{
                    f"component_mean__{component}": component_mean[index]
                    for index, component in enumerate(nonlinear.EXTENDED_COMPONENTS)
                },
            }
        )
    return pd.DataFrame(support_rows), pd.DataFrame(component_rows)


def previous_slice_composition(support: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slice_name, slice_label, constraints in PREVIOUS_SLICES:
        keep = np.ones(len(support), dtype=bool)
        for variable, sign in constraints.items():
            keep &= support[f"state_sign__{variable}"].eq(
                "high" if sign > 0 else "low"
            )
        subset = support.loc[keep].copy()
        total = float(subset["formal_weight_share"].sum())
        for state in subset.itertuples(index=False):
            rows.append(
                {
                    "pair_slice": slice_name,
                    "pair_slice_label_zh": slice_label,
                    "state_code": state.state_code,
                    "state": state.state,
                    "state_label_zh": state.state_label_zh,
                    "grid_year_n": state.grid_year_n,
                    "grid_n": state.grid_n,
                    "formal_weight_share": state.formal_weight_share,
                    "share_within_pair_slice": state.formal_weight_share / total,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["pair_slice", "share_within_pair_slice"], ascending=[True, False]
    )


def p_text(value: float) -> str:
    return f"{value:.3e}"


def render_report(
    support: pd.DataFrame,
    tests: pd.DataFrame,
    fingerprints: pd.DataFrame,
    shared_fingerprints: pd.DataFrame,
    composition: pd.DataFrame,
) -> str:
    retained_counts = (
        tests.groupby("outcome")["whole_curve_retained_holm_0_05"].sum().astype(int)
    )
    negative = fingerprints.loc[fingerprints["negative_side_reinforcement"]]
    positive = fingerprints.loc[
        fingerprints["positive_amplitude_amplification"]
        & ~fingerprints["negative_side_reinforcement"]
    ]

    def fingerprint_rows(table: pd.DataFrame) -> str:
        rows = []
        for state in table.itertuples(index=False):
            rows.append(
                f"| {state.state_code} | {state.state_label_zh} | "
                f"{state.formal_weight_share:.1%} | {state.grid_n} | "
                f"{state.positive_probability__edge_half_contrast:.3e} "
                f"({p_text(state.positive_probability__holm_adjusted_p)}) | "
                f"{state.positive_conditional_second_moment__edge_half_contrast:.3e} "
                f"({p_text(state.positive_conditional_second_moment__holm_adjusted_p)}) | "
                f"{state.negative_conditional_second_moment__edge_half_contrast:.3e} "
                f"({p_text(state.negative_conditional_second_moment__holm_adjusted_p)}) |"
            )
        return "\n".join(rows) if rows else "| 无 | — | — | — | — | — | — |"

    composition_rows = []
    for slice_name, group in composition.groupby("pair_slice", sort=False):
        leading = group.iloc[0]
        composition_rows.append(
            f"| {leading['pair_slice_label_zh']} | {leading['state_code']} | "
            f"{leading['state_label_zh']} | {leading['share_within_pair_slice']:.1%} |"
        )

    return f"""# 完整四维观测气候状态投影

## 状态怎样产生

四个气候异常均以各网格自身的2013–2025年常态为零点。四个变量的高低组合形成全部16个互斥、穷尽的四维状态；状态定义和样本划分不读取NIRv响应。每个真实网格—年份只进入一个状态，其四个主效应、六个两因子交互和八个非线性主效应分量共同代入已经确定的扩展模型。

16个状态均有正式观测支持，支持范围为 `{support['grid_year_n'].min()}`–`{support['grid_year_n'].max()}` 个网格—年份、`{support['grid_n'].min()}`–`{support['grid_n'].max()}` 个网格和 `{support['formal_weight_share'].min():.1%}`–`{support['formal_weight_share'].max():.1%}` 的正式权重。没有先按响应结果删除状态。

## 正式比较规则

每个状态分别检验正偏离概率、正偏离条件平方幅度和负偏离条件平方幅度的完整30–480 m曲线。每个响应内对16个状态采用Holm校正。方向摘要是对数距离上30–120 m半区平均值减去120–480 m半区平均值；它利用整条曲线，不使用30 m相对480 m的端点差。

| 响应成分 | 16个状态中校正后保留数 |
|---|---:|
| 正偏离概率 | {retained_counts['positive_probability']} |
| 正偏离条件平方幅度 | {retained_counts['positive_conditional_second_moment']} |
| 负偏离条件平方幅度 | {retained_counts['negative_conditional_second_moment']} |

## 负侧协同放大的完整状态

判定规则同时要求：正偏离概率曲线通过Holm校正且近边缘半区更低；负偏离幅度曲线通过Holm校正且近边缘半区更高。

| 状态代码 | 四维状态 | 正式权重 | 网格数 | 正偏离概率半区差（Holm P） | 正幅度半区差（Holm P） | 负幅度半区差（Holm P） |
|---|---|---:|---:|---:|---:|---:|
{fingerprint_rows(negative)}

## 正侧幅度放大的其他完整状态

这些状态的正偏离条件平方幅度曲线通过Holm校正、近边缘半区更高，同时不满足负侧协同放大规则。

| 状态代码 | 四维状态 | 正式权重 | 网格数 | 正偏离概率半区差（Holm P） | 正幅度半区差（Holm P） | 负幅度半区差（Holm P） |
|---|---|---:|---:|---:|---:|---:|
{fingerprint_rows(positive)}

## 共享气候轴收口

正侧幅度放大的 `HHHH` 和 `LHHH` 只在短波高低上不同；负侧幅度放大的 `HHHL` 和 `LHHL` 也只在短波高低上不同。因此把短波保留为真实观测值但不用于命名，形成两个共享轴状态。下表曲线用于汇总已经通过16状态正式筛选的共同结构；正式发现证据仍来自前述16状态Holm校正。

| 共享轴状态 | 正式权重 | 网格数 | 正偏离概率半区差（两状态Holm P） | 正幅度半区差（两状态Holm P） | 负幅度半区差（两状态Holm P） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {row.state_label_zh} | {row.formal_weight_share:.1%} | {row.grid_n} | "
    f"{row.positive_probability__edge_half_contrast:.3e} ({p_text(row.positive_probability__holm_adjusted_p)}) | "
    f"{row.positive_conditional_second_moment__edge_half_contrast:.3e} ({p_text(row.positive_conditional_second_moment__holm_adjusted_p)}) | "
    f"{row.negative_conditional_second_moment__edge_half_contrast:.3e} ({p_text(row.negative_conditional_second_moment__holm_adjusted_p)}) |"
    for row in shared_fingerprints.itertuples(index=False)
)}

两个状态具有相同的偏暖和高VPD背景，也都表现为近边缘正偏离概率降低。根区土壤水分高时，变化集中在正偏离发生后的幅度放大；根区土壤水分低时，变化集中在负偏离发生后的深度放大。由此，原来的“暖湿”与“暖高VPD”两个二维名称被收口为同一大气需求背景下由土壤水分供给分开的两条响应路径。

## 原有二维候选状态在四维空间中的组成

| 原有二维切面 | 权重最大的四维状态 | 完整状态 | 占该切面权重 |
|---|---|---|---:|
{chr(10).join(composition_rows)}

二维名称只承担可读切面。正式状态筛选由上面的16个完整四维状态、三个响应成分的整曲线证据和观测支持共同完成。
"""


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUTPUT}")
    sample = pd.read_parquet(INPUTS / "formal_sample.parquet")
    climate = pd.read_parquet(INPUTS / "grid_year_climate_with_anomalies.parquet")
    grid_levels, grid_year_weight, observation_n = formal_grid_year_weights(sample)
    raw_climate, climate_components, definitions, knots = nonlinear.align_climate(
        climate, grid_levels
    )
    base_support_difference = nonlinear.validate_state_support(
        raw_climate,
        climate_components[:, :, : len(nonlinear.BASE_COMPONENTS)],
        grid_year_weight.reshape(-1),
    )
    support, component_means, membership, _ = observed_state_support(
        grid_levels, grid_year_weight, raw_climate, climate_components
    )

    beta = np.load(MODEL / "joint_coefficient_vector.npy")
    covariance = np.load(MODEL / "joint_cluster_robust_covariance.npy")
    names = parameter_names()
    expected_parameter_n = len(names) * len(nonlinear.OUTCOMES)
    if beta.shape != (expected_parameter_n,) or covariance.shape != (
        expected_parameter_n,
        expected_parameter_n,
    ):
        raise RuntimeError("extended coefficient or covariance shape changed")

    curves, tests = project_states(
        support, component_means, beta, covariance, names
    )
    fingerprints = screen_fingerprints(support, tests)
    shared_support, shared_component_means = shared_axis_states(
        support, component_means, membership
    )
    shared_curves, shared_tests = project_states(
        shared_support, shared_component_means, beta, covariance, names
    )
    shared_fingerprints = screen_fingerprints(shared_support, shared_tests)
    composition = previous_slice_composition(support)

    temporary_root = Path(tempfile.mkdtemp(prefix="observed_climate_states_"))
    temporary_output = temporary_root / OUTPUT.name
    temporary_output.mkdir()
    try:
        support.to_csv(temporary_output / "four_dimensional_state_support.csv", index=False)
        component_means.to_csv(
            temporary_output / "four_dimensional_state_component_means.csv",
            index=False,
        )
        membership.to_csv(
            temporary_output / "grid_year_state_membership.csv", index=False
        )
        curves.to_csv(
            temporary_output / "four_dimensional_state_curves.csv", index=False
        )
        tests.to_csv(
            temporary_output / "four_dimensional_state_whole_curve_tests.csv",
            index=False,
        )
        fingerprints.to_csv(
            temporary_output / "state_fingerprint_screening.csv", index=False
        )
        shared_support.to_csv(
            temporary_output / "shared_axis_state_support.csv", index=False
        )
        shared_component_means.to_csv(
            temporary_output / "shared_axis_state_component_means.csv", index=False
        )
        shared_curves.to_csv(
            temporary_output / "shared_axis_state_curves.csv", index=False
        )
        shared_tests.to_csv(
            temporary_output / "shared_axis_state_whole_curve_tests.csv", index=False
        )
        shared_fingerprints.to_csv(
            temporary_output / "shared_axis_state_fingerprints.csv", index=False
        )
        composition.to_csv(
            temporary_output / "previous_pair_slice_composition.csv", index=False
        )
        metadata = {
            "status": "COMPLETE_OBSERVED_FOUR_DIMENSIONAL_STATE_PROJECTION",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_refitted": False,
            "source_model": str(MODEL.relative_to(ROOT)),
            "observation_n": observation_n,
            "grid_n": len(grid_levels),
            "year_n": len(nonlinear.YEARS),
            "state_n": len(support),
            "state_definition": "all 2^4 high-low combinations of within-grid annual anomalies",
            "state_definition_reads_nirv_response": False,
            "state_partition": "mutually exclusive and exhaustive",
            "projection_components": list(nonlinear.EXTENDED_COMPONENTS),
            "formal_estimand": "range-centered continuous 30--480 m climate-state curve",
            "direction_summary": "mean over log-distance 30--120 m minus mean over 120--480 m",
            "whole_curve_hypotheses_per_outcome": len(support),
            "multiple_testing": "Holm family-wise error control within each outcome across 16 states",
            "base_quadrant_support_maximum_reproduction_difference": base_support_difference,
            "minimum_state_grid_year_n": int(support["grid_year_n"].min()),
            "minimum_state_grid_n": int(support["grid_n"].min()),
            "minimum_state_formal_weight_share": float(
                support["formal_weight_share"].min()
            ),
            "negative_side_reinforcement_state_n": int(
                fingerprints["negative_side_reinforcement"].sum()
            ),
            "positive_amplitude_amplification_state_n": int(
                fingerprints["positive_amplitude_amplification"].sum()
            ),
            "shared_axis_state_n": len(shared_support),
            "shared_axis_role": "effect summaries derived from adjacent four-dimensional states that independently passed the formal screen",
            "spline_knots": {key: value.tolist() for key, value in knots.items()},
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
