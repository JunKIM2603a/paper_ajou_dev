from __future__ import annotations

from collections.abc import Mapping, Sequence


def split_group_ids(
    group_labels: Mapping[str, int],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")

    group_ids = list(group_labels.keys())
    labels = [int(group_labels[group_id]) for group_id in group_ids]
    train_val_ids, test_ids = _train_test_group_split(
        group_ids,
        labels,
        test_size=test_ratio,
        random_state=seed,
    )
    train_val_labels = [int(group_labels[group_id]) for group_id in train_val_ids]
    train_ids, val_ids = _train_test_group_split(
        train_val_ids,
        train_val_labels,
        test_size=validation_ratio,
        random_state=seed + 100,
    )
    return {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }


def _train_test_group_split(
    group_ids: Sequence[str],
    labels: Sequence[int],
    *,
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    from sklearn.model_selection import train_test_split

    if len(group_ids) == 0:
        return [], []

    stratify = labels if _can_stratify(labels) else None
    try:
        train_ids, test_ids = train_test_split(
            list(group_ids),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train_ids, test_ids = train_test_split(
            list(group_ids),
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return list(train_ids), list(test_ids)


def _can_stratify(labels: Sequence[int]) -> bool:
    if len(labels) < 2:
        return False
    unique_values = set(labels)
    if len(unique_values) < 2:
        return False
    for value in unique_values:
        if sum(1 for label in labels if label == value) < 2:
            return False
    return True
