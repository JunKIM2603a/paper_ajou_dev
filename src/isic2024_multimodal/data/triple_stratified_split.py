from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TripleSplitResult:
    patient_assignment: dict[str, int]
    patient_profile: Any
    balance_score: float


@dataclass(frozen=True)
class NestedCVResult:
    outer: TripleSplitResult
    inner_by_outer_fold: dict[int, TripleSplitResult]
    patient_profile: Any
    outer_folds: int
    inner_folds: int


def make_patient_split_profile(
    frame,
    *,
    patient_column: str = "patient_id",
    lesion_column: str = "lesion_id",
    sample_column: str = "isic_id",
    target_column: str = "target",
    sample_count_bins: int = 5,
):
    """Build patient-level rows used by the split balancing objective."""
    import pandas as pd

    required_columns = [patient_column, sample_column, target_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing required split columns: {missing_columns}")

    aggregation = {
        "patient_rows": (sample_column, "size"),
        "positive_rows": (target_column, "sum"),
    }
    if lesion_column in frame.columns:
        aggregation["lesion_count"] = (lesion_column, lambda value: value.dropna().nunique())
    else:
        aggregation["lesion_count"] = (sample_column, "size")

    profile = frame.groupby(patient_column, dropna=False).agg(**aggregation).reset_index()
    if profile[patient_column].isna().any():
        raise ValueError("patient_id contains missing values; patient-level split cannot be paper-valid.")

    profile[patient_column] = profile[patient_column].astype(str)
    profile["positive_rows"] = profile["positive_rows"].astype(int)
    profile["has_malignant"] = profile["positive_rows"] > 0
    profile["sample_count_bin"] = _make_sample_count_bin(profile["patient_rows"], q=sample_count_bins)
    profile["positive_row_bin"] = np.select(
        [profile["positive_rows"].eq(0), profile["positive_rows"].eq(1)],
        ["zero", "one"],
        default="multiple",
    )
    profile["triple_stratum"] = (
        profile["has_malignant"].astype(int).astype(str)
        + "|pos="
        + profile["positive_row_bin"].astype(str)
        + "|size="
        + profile["sample_count_bin"].astype(str)
    )
    return profile


def assign_triple_stratified_groups(
    patient_profile,
    *,
    n_groups: int,
    target_ratios: list[float],
    seed: int,
    max_local_search_passes: int = 3,
) -> TripleSplitResult:
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if len(target_ratios) != n_groups:
        raise ValueError("target_ratios length must match n_groups")
    if not np.isclose(sum(target_ratios), 1.0):
        raise ValueError(f"target_ratios must sum to 1.0, got {target_ratios}")

    profile = patient_profile.copy().reset_index(drop=True)
    patient_column = "patient_id"
    bin_values = sorted(profile["sample_count_bin"].astype(str).unique().tolist())
    profile["sample_count_bin"] = profile["sample_count_bin"].astype(str)

    total = _summarize_total(profile, bin_values)
    assignment = _build_initial_quota_assignment(
        profile,
        patient_column=patient_column,
        n_groups=n_groups,
        target_ratios=target_ratios,
        seed=seed,
    )
    groups = _build_group_state_from_assignment(profile, assignment, n_groups, bin_values)
    best_score = _triple_balance_score(groups, total, target_ratios, bin_values)

    patient_rows_by_id = {
        str(patient_row["patient_id"]): patient_row for _, patient_row in profile.iterrows()
    }
    patient_ids = sorted(assignment)

    for _ in range(max_local_search_passes):
        improved = False
        for left_index, left_patient_id in enumerate(patient_ids):
            for right_patient_id in patient_ids[left_index + 1 :]:
                left_group = assignment[left_patient_id]
                right_group = assignment[right_patient_id]
                if left_group == right_group:
                    continue

                left_row = patient_rows_by_id[left_patient_id]
                right_row = patient_rows_by_id[right_patient_id]
                candidate_groups = _copy_group_state_for_swap(groups, left_group, right_group)
                _move_patient(
                    candidate_groups,
                    left_row,
                    from_group=left_group,
                    to_group=right_group,
                    track_patient_ids=False,
                )
                _move_patient(
                    candidate_groups,
                    right_row,
                    from_group=right_group,
                    to_group=left_group,
                    track_patient_ids=False,
                )
                candidate_score = _triple_balance_score(candidate_groups, total, target_ratios, bin_values)

                if candidate_score + 1e-12 < best_score:
                    assignment[left_patient_id] = right_group
                    assignment[right_patient_id] = left_group
                    groups = candidate_groups
                    best_score = candidate_score
                    improved = True
        if not improved:
            break

    return TripleSplitResult(patient_assignment=assignment, patient_profile=profile, balance_score=best_score)


def build_holdout_and_cv_assignments(
    frame,
    *,
    seed: int = 42,
    test_size: float = 0.20,
    cv_folds: int = 5,
    sample_count_bins: int = 5,
) -> dict[str, Any]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")

    patient_profile = make_patient_split_profile(frame, sample_count_bins=sample_count_bins)
    holdout_result = assign_triple_stratified_groups(
        patient_profile,
        n_groups=2,
        target_ratios=[1.0 - test_size, test_size],
        seed=seed,
    )
    train_validation_patients = {
        patient_id for patient_id, group_index in holdout_result.patient_assignment.items() if group_index == 0
    }
    train_validation_profile = patient_profile.loc[
        patient_profile["patient_id"].isin(train_validation_patients)
    ].copy()
    cv_result = assign_triple_stratified_groups(
        train_validation_profile,
        n_groups=cv_folds,
        target_ratios=[1.0 / cv_folds] * cv_folds,
        seed=seed + 1000,
    )
    return {
        "holdout": holdout_result,
        "cv": cv_result,
    }


def build_nested_cv_assignments(
    frame,
    *,
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 4,
    sample_count_bins: int = 5,
) -> NestedCVResult:
    """Build patient-level Triple Stratified outer and inner CV assignments.

    The outer assignment defines ``cv_test_fold`` / ``outer_test``. For each
    outer fold, the remaining patients are split again with the same balancing
    objective to produce inner validation folds. The inner assignment is scoped
    to the corresponding outer fold and must never include outer-test patients.
    """
    if outer_folds < 2:
        raise ValueError(f"outer_folds must be >= 2, got {outer_folds}")
    if inner_folds < 2:
        raise ValueError(f"inner_folds must be >= 2, got {inner_folds}")
    if inner_folds >= outer_folds:
        raise ValueError(
            "inner_folds must be smaller than outer_folds so each outer train "
            f"pool can be re-split without reusing the outer test fold, got "
            f"inner_folds={inner_folds}, outer_folds={outer_folds}"
        )

    patient_profile = make_patient_split_profile(frame, sample_count_bins=sample_count_bins)
    outer_result = assign_triple_stratified_groups(
        patient_profile,
        n_groups=outer_folds,
        target_ratios=[1.0 / outer_folds] * outer_folds,
        seed=seed,
    )

    inner_by_outer_fold: dict[int, TripleSplitResult] = {}
    for outer_fold in range(outer_folds):
        outer_test_patients = {
            patient_id
            for patient_id, assigned_outer_fold in outer_result.patient_assignment.items()
            if int(assigned_outer_fold) == outer_fold
        }
        outer_train_profile = patient_profile.loc[
            ~patient_profile["patient_id"].astype(str).isin(outer_test_patients)
        ].copy()
        inner_by_outer_fold[outer_fold] = assign_triple_stratified_groups(
            outer_train_profile,
            n_groups=inner_folds,
            target_ratios=[1.0 / inner_folds] * inner_folds,
            seed=seed + 1000 + outer_fold,
        )

    return NestedCVResult(
        outer=outer_result,
        inner_by_outer_fold=inner_by_outer_fold,
        patient_profile=patient_profile,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
    )


def _make_sample_count_bin(series, *, q: int):
    import pandas as pd

    if q < 2:
        raise ValueError(f"sample_count_bins must be >= 2, got {q}")
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype(int).astype(str)


def _build_initial_quota_assignment(patient_frame, *, patient_column: str, n_groups: int, target_ratios: list[float], seed: int):
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for _, stratum_frame in patient_frame.groupby("triple_stratum", sort=True):
        shuffled = stratum_frame.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        expected = np.array(target_ratios, dtype=float) * len(shuffled)
        quotas = np.floor(expected).astype(int)
        remainder = len(shuffled) - int(quotas.sum())
        if remainder > 0:
            fractional_order = np.argsort(-(expected - quotas))
            for group_index in fractional_order[:remainder]:
                quotas[group_index] += 1

        cursor = 0
        for group_index, quota in enumerate(quotas):
            for patient_id in shuffled.iloc[cursor : cursor + quota][patient_column]:
                assignment[str(patient_id)] = group_index
            cursor += quota
    _ensure_non_empty_groups(assignment, n_groups)
    return assignment


def _ensure_non_empty_groups(assignment: dict[str, int], n_groups: int) -> None:
    if len(assignment) < n_groups:
        return

    for group_index in range(n_groups):
        if group_index in assignment.values():
            continue
        counts = {candidate_group: list(assignment.values()).count(candidate_group) for candidate_group in range(n_groups)}
        donor_group = max(counts, key=lambda candidate_group: counts[candidate_group])
        donor_patients = sorted(patient_id for patient_id, assigned_group in assignment.items() if assigned_group == donor_group)
        if len(donor_patients) <= 1:
            continue
        assignment[donor_patients[-1]] = group_index


def _empty_group_state(bin_values: list[str]) -> dict[str, Any]:
    return {
        "patient_ids": [],
        "patients": 0,
        "rows": 0.0,
        "positive_rows": 0.0,
        "malignant_patients": 0.0,
        "bin_counts": {bin_value: 0.0 for bin_value in bin_values},
    }


def _build_group_state_from_assignment(patient_frame, assignment: dict[str, int], n_groups: int, bin_values: list[str]):
    groups = [_empty_group_state(bin_values) for _ in range(n_groups)]
    for _, patient_row in patient_frame.iterrows():
        _add_patient(groups[int(assignment[str(patient_row["patient_id"])])], patient_row)
    return groups


def _add_patient(state: dict[str, Any], patient_row, *, track_patient_ids: bool = True) -> None:
    if track_patient_ids:
        state["patient_ids"].append(str(patient_row["patient_id"]))
    state["patients"] += 1
    state["rows"] += float(patient_row["patient_rows"])
    state["positive_rows"] += float(patient_row["positive_rows"])
    state["malignant_patients"] += float(patient_row["has_malignant"])
    state["bin_counts"][str(patient_row["sample_count_bin"])] += 1.0


def _remove_patient(state: dict[str, Any], patient_row, *, track_patient_ids: bool = True) -> None:
    if track_patient_ids:
        state["patient_ids"].remove(str(patient_row["patient_id"]))
    state["patients"] -= 1
    state["rows"] -= float(patient_row["patient_rows"])
    state["positive_rows"] -= float(patient_row["positive_rows"])
    state["malignant_patients"] -= float(patient_row["has_malignant"])
    state["bin_counts"][str(patient_row["sample_count_bin"])] -= 1.0


def _move_patient(groups, patient_row, *, from_group: int, to_group: int, track_patient_ids: bool = True) -> None:
    _remove_patient(groups[from_group], patient_row, track_patient_ids=track_patient_ids)
    _add_patient(groups[to_group], patient_row, track_patient_ids=track_patient_ids)


def _copy_group_state(groups, *, include_patient_ids: bool = True):
    return [
        _copy_one_group_state(state, include_patient_ids=include_patient_ids)
        for state in groups
    ]


def _copy_group_state_for_swap(groups, first_group: int, second_group: int):
    copied_groups = list(groups)
    copied_groups[first_group] = _copy_one_group_state(groups[first_group], include_patient_ids=False)
    copied_groups[second_group] = _copy_one_group_state(groups[second_group], include_patient_ids=False)
    return copied_groups


def _copy_one_group_state(state, *, include_patient_ids: bool):
    return {
        "patient_ids": list(state["patient_ids"]) if include_patient_ids else [],
        "patients": state["patients"],
        "rows": state["rows"],
        "positive_rows": state["positive_rows"],
        "malignant_patients": state["malignant_patients"],
        "bin_counts": dict(state["bin_counts"]),
    }


def _summarize_total(patient_frame, bin_values: list[str]) -> dict[str, Any]:
    return {
        "patients": float(len(patient_frame)),
        "rows": float(patient_frame["patient_rows"].sum()),
        "positive_rows": float(patient_frame["positive_rows"].sum()),
        "malignant_patients": float(patient_frame["has_malignant"].sum()),
        "bin_counts": {
            bin_value: float(patient_frame["sample_count_bin"].astype(str).eq(bin_value).sum()) for bin_value in bin_values
        },
    }


def _relative_abs_error(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1.0)


def _triple_balance_score(groups, total: dict[str, Any], target_ratios: list[float], bin_values: list[str]) -> float:
    score = 0.0
    for group_index, state in enumerate(groups):
        target_ratio = target_ratios[group_index]
        score += 4.0 * _relative_abs_error(state["patients"], total["patients"] * target_ratio)
        score += 6.0 * _relative_abs_error(state["rows"], total["rows"] * target_ratio)
        score += 10.0 * _relative_abs_error(state["positive_rows"], total["positive_rows"] * target_ratio)
        score += 8.0 * _relative_abs_error(state["malignant_patients"], total["malignant_patients"] * target_ratio)
        for bin_value in bin_values:
            score += 3.0 * _relative_abs_error(
                state["bin_counts"][bin_value],
                total["bin_counts"][bin_value] * target_ratio,
            )
    return float(score)
