from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from isic2024_multimodal.evaluation.metrics import (
    PRIMARY_PAUC_METRIC,
    select_threshold_by_f1,
    thresholded_binary_classification_metrics,
)


def run_training(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    device: str,
    hyperparameters: dict[str, Any],
    output_dir: str | Path,
    mlflow_client: Any | None = None,
) -> dict[str, Any]:
    # 한 trial의 체크포인트, 히스토리, 요약 JSON을 모두 같은 폴더에 모은다.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"trial output dir: {output_dir}")

    criterion = nn.CrossEntropyLoss()
    optimizer_name = hyperparameters.get("optimizer", "adamw").lower()
    learning_rate = float(hyperparameters.get("learning_rate", 1e-4))
    weight_decay = float(hyperparameters.get("weight_decay", 1e-4))
    epochs = int(hyperparameters.get("epochs", 5))
    _log(
        "training setup: "
        f"optimizer={optimizer_name}, lr={learning_rate}, weight_decay={weight_decay}, epochs={epochs}, device={device}"
    )
    _log(
        "dataloader sizes: "
        f"train={len(dataloaders['train'].dataset)}, val={len(dataloaders['val'].dataset)}, test={len(dataloaders['test'].dataset)}"
    )

    # optimizer는 search_space에서 넘어온 설정값으로 선택한다.
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(hyperparameters.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    history: list[dict[str, Any]] = []
    best_state = None
    best_metric = float("-inf")
    best_epoch = -1
    best_validation_metrics: dict[str, float] | None = None
    best_validation_threshold = 0.5
    started = time.time()

    # 매 epoch마다 train loss와 val metric을 기록하고 최고 성능 가중치를 따로 보관한다.
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        _log(f"epoch {epoch}/{epochs}: train start")
        train_loss = _train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        _log(f"epoch {epoch}/{epochs}: train done, loss={train_loss:.6f}")
        scheduler.step()
        _log(f"epoch {epoch}/{epochs}: validation start")
        val_labels, val_probabilities = collect_model_outputs(model, dataloaders["val"], device)
        val_threshold = select_threshold_by_f1(val_labels, val_probabilities)
        val_metrics = evaluate_outputs(
            val_labels,
            val_probabilities,
            threshold=val_threshold,
        )
        _log(
            "epoch "
            f"{epoch}/{epochs}: validation done, "
            f"acc={val_metrics['accuracy']:.4f}, f1={val_metrics['f1_score']:.4f}, "
            f"auc={val_metrics['auc_roc']:.4f}, {PRIMARY_PAUC_METRIC}={val_metrics[PRIMARY_PAUC_METRIC]:.4f}, "
            f"threshold={val_threshold:.6f}"
        )
        epoch_duration_seconds = time.time() - epoch_started
        estimated_remaining_seconds = epoch_duration_seconds * max(epochs - epoch, 0)
        _log(
            f"epoch {epoch}/{epochs}: duration={epoch_duration_seconds:.1f}s, "
            f"estimated_remaining={format_duration(estimated_remaining_seconds)}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_selected_threshold": val_threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_duration_seconds": epoch_duration_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
        }
        history.append(row)

        if mlflow_client is not None:
            for key, value in row.items():
                if key == "epoch":
                    continue
                mlflow_client.log_metric(key, float(value), step=epoch)

        score = val_metrics[PRIMARY_PAUC_METRIC]
        if score != score:
            score = val_metrics["auc_roc"]
        if score != score:
            score = val_metrics["f1_score"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_validation_metrics = dict(val_metrics)
            best_validation_threshold = val_threshold
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            _log(f"epoch {epoch}/{epochs}: new best model saved in memory")

    # 테스트는 마지막 epoch가 아니라 검증 기준 최고 성능 가중치로 수행한다.
    if best_state is not None:
        model.load_state_dict(best_state)

    _log("best validation evaluation start")
    best_val_labels, best_val_probabilities = collect_model_outputs(model, dataloaders["val"], device)
    selected_threshold = select_threshold_by_f1(best_val_labels, best_val_probabilities)
    best_validation_metrics = evaluate_outputs(
        best_val_labels,
        best_val_probabilities,
        threshold=selected_threshold,
    )
    best_validation_threshold = selected_threshold
    _log(f"selected threshold from validation_f1: {selected_threshold:.6f}")

    _log("test evaluation start")
    test_labels, test_probabilities = collect_model_outputs(model, dataloaders["test"], device)
    test_metrics = evaluate_outputs(
        test_labels,
        test_probabilities,
        threshold=selected_threshold,
    )
    _log(
        "test evaluation done, "
        f"acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1_score']:.4f}, "
        f"auc={test_metrics['auc_roc']:.4f}, {PRIMARY_PAUC_METRIC}={test_metrics[PRIMARY_PAUC_METRIC]:.4f}, "
        f"threshold={selected_threshold:.6f}"
    )
    model_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    history_path = output_dir / "history.csv"
    _write_history_csv(history, history_path)
    summary = {
        "best_epoch": best_epoch,
        "primary_validation_metric_name": PRIMARY_PAUC_METRIC,
        "best_validation_metric": best_metric,
        "best_validation_metrics": best_validation_metrics or {},
        "threshold_source": "validation_f1",
        "selected_threshold": best_validation_threshold,
        "test_metrics": test_metrics,
        "duration_seconds": time.time() - started,
        "last_epoch_duration_seconds": history[-1].get("epoch_duration_seconds") if history else None,
        "last_estimated_remaining_seconds": history[-1].get("estimated_remaining_seconds") if history else None,
        "model_path": str(model_path),
        "history_path": str(history_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    _log(f"trial complete in {summary['duration_seconds']:.1f}s")
    return summary


# malignant 확률을 모은 뒤 validation에서 선택한 threshold로 threshold-dependent metric을 계산한다.
def collect_model_outputs(model: nn.Module, dataloader: DataLoader, device: str) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    logged_first_batch = False

    with torch.no_grad():
        for inputs, targets in dataloader:
            if not logged_first_batch:
                _log(
                    "evaluation first batch: "
                    f"batch_size={inputs.size(0)}, image_shape={tuple(inputs.shape)}, device={device}"
                )
                logged_first_batch = True
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]

            labels.extend(targets.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    return labels, probabilities


def evaluate_outputs(labels: list[int], probabilities: list[float], *, threshold: float) -> dict[str, float]:
    return thresholded_binary_classification_metrics(labels, probabilities, threshold=threshold)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    labels, probabilities = collect_model_outputs(model, dataloader, device)
    selected_threshold = select_threshold_by_f1(labels, probabilities) if threshold is None else threshold
    return evaluate_outputs(labels, probabilities, threshold=selected_threshold)


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    logged_first_batch = False

    for inputs, targets in dataloader:
        if not logged_first_batch:
            _log(
                "train first batch: "
                f"batch_size={inputs.size(0)}, image_shape={tuple(inputs.shape)}, device={device}"
            )
            logged_first_batch = True
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[trainer {timestamp}] {message}", flush=True)




