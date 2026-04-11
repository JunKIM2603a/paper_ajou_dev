from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve as sklearn_roc_curve


PRIMARY_MIN_TPR = 0.80
PRIMARY_PAUC_METRIC = f"pauc_above_tpr{int(round(PRIMARY_MIN_TPR * 100)):02d}"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def pauc_metric_name(min_tpr: float) -> str:
    return f"pauc_above_tpr{int(round(min_tpr * 100)):02d}"


def binary_classification_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
    *,
    min_tpr: float = PRIMARY_MIN_TPR,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for label, pred in zip(labels, predictions, strict=True):
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1

    accuracy = _safe_div(tp + tn, len(labels))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0
    auc = roc_auc_score(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)
    pauc = partial_auc_above_min_tpr(labels, probabilities, min_tpr=min_tpr)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "auc_roc": auc,
        "average_precision": average_precision,
        pauc_metric_name(min_tpr): pauc,
    }


def roc_auc_score(labels: list[int], probabilities: list[float]) -> float:
    positives = sum(1 for value in labels if value == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    current_rank = 1
    index = 0
    while index < len(ranked):
        next_index = index + 1
        while next_index < len(ranked) and ranked[next_index][0] == ranked[index][0]:
            next_index += 1

        average_rank = (current_rank + (current_rank + (next_index - index) - 1)) / 2.0
        positives_in_tie = sum(label for _, label in ranked[index:next_index])
        rank_sum += positives_in_tie * average_rank
        current_rank += next_index - index
        index = next_index

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision_score(labels: list[int], probabilities: list[float]) -> float:
    positives = sum(1 for value in labels if value == 1)
    if positives == 0:
        return math.nan

    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0

    for index, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / index

    return precision_sum / positives


def partial_auc_above_min_tpr(
    labels: list[int],
    probabilities: list[float],
    *,
    min_tpr: float = PRIMARY_MIN_TPR,
) -> float:
    positives = sum(1 for value in labels if value == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    if not 0 < min_tpr < 1:
        raise ValueError(f"Expected min_tpr in range (0, 1), got: {min_tpr!r}")

    max_fpr = 1.0 - min_tpr
    flipped_labels = 1 - np.asarray(labels, dtype=int)
    # The official ISIC implementation flips scores with `1 - score`.
    # Using `-score` keeps the ordering identical even when the input scores are not probabilities.
    flipped_scores = -np.asarray(probabilities, dtype=float)
    fpr, tpr, _ = sklearn_roc_curve(flipped_labels, flipped_scores)

    if max_fpr == 1.0:
        return float(sklearn_auc(fpr, tpr))

    stop = np.searchsorted(fpr, max_fpr, side="right")
    if stop <= 0:
        return 0.0
    if stop >= len(fpr):
        return float(sklearn_auc(fpr, tpr))

    interpolated_tpr = np.interp(
        max_fpr,
        [fpr[stop - 1], fpr[stop]],
        [tpr[stop - 1], tpr[stop]],
    )
    fpr = np.append(fpr[:stop], max_fpr)
    tpr = np.append(tpr[:stop], interpolated_tpr)
    return float(sklearn_auc(fpr, tpr))
