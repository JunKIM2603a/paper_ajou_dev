from __future__ import annotations

import math


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_classification_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
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
    f1 = _safe_div(2 * precision * recall, precision + recall)
    auc = roc_auc_score(labels, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
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

