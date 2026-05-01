from __future__ import annotations

from typing import Any

import numpy as np


class _LinearBinaryModel:
    def __init__(self, input_dim: int):
        import torch.nn as nn

        self.module = nn.Linear(input_dim, 1)

    def __call__(self, x):
        return self.module(x).squeeze(1)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def to(self, device):
        self.module.to(device)
        return self


class _MLPBinaryModel:
    def __init__(self, input_dim: int, hidden_layer_sizes: tuple[int, ...] | list[int]):
        import torch.nn as nn

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, 1))
        self.module = nn.Sequential(*layers)

    def __call__(self, x):
        return self.module(x).squeeze(1)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def to(self, device):
        self.module.to(device)
        return self


class TorchTabularEstimator:
    def __init__(
        self,
        *,
        model_name: str,
        preprocessor,
        hyperparameters: dict[str, Any],
        device: str,
        scale_pos_weight: float,
    ) -> None:
        self.model_name = model_name
        self.preprocessor = preprocessor
        self.hyperparameters = dict(hyperparameters)
        self.device = device
        self.scale_pos_weight = float(scale_pos_weight)
        self.model = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        seed = int(self.hyperparameters["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        features = self._transform(X, fit=True)
        targets = np.asarray(y, dtype=np.float32)
        input_dim = int(features.shape[1])

        torch_device = torch.device(self.device)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(feature_tensor, target_tensor)
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = DataLoader(
            dataset,
            batch_size=self._train_batch_size(),
            shuffle=True,
            generator=generator,
            pin_memory=torch_device.type == "cuda",
        )

        self.model = self._build_model(input_dim).to(torch_device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._learning_rate(),
            weight_decay=self._weight_decay(len(targets)),
        )
        pos_weight = torch.tensor(self.scale_pos_weight, dtype=torch.float32, device=torch_device)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        patience = self._patience()

        for _epoch in range(self._max_epochs()):
            self.model.train()
            epoch_loss = 0.0
            seen_rows = 0
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(torch_device, non_blocking=True)
                batch_targets = batch_targets.to(torch_device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_features)
                if self.model_name == "svm":
                    signed_targets = batch_targets.mul(2.0).sub(1.0)
                    hinge = torch.clamp(1.0 - signed_targets * logits, min=0.0)
                    loss = hinge.square().mean() * self._svm_c_scale()
                else:
                    loss = bce_criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

                batch_size = int(batch_targets.shape[0])
                epoch_loss += float(loss.detach().cpu()) * batch_size
                seen_rows += batch_size

            current_loss = epoch_loss / max(seen_rows, 1)
            if current_loss + 1e-6 < best_loss:
                best_loss = current_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        scores = self._predict_scores(X)
        return (scores >= 0.5).astype(int)

    def predict_proba(self, X):
        scores = self._predict_scores(X)
        return np.column_stack([1.0 - scores, scores])

    def decision_function(self, X):
        return self._predict_logits(X)

    def _transform(self, X, *, fit: bool) -> np.ndarray:
        matrix = self.preprocessor.fit_transform(X) if fit else self.preprocessor.transform(X)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def _predict_logits(self, X) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("TorchTabularEstimator must be fitted before prediction.")

        matrix = self._transform(X, fit=False)
        torch_device = torch.device(self.device)
        logits_chunks: list[np.ndarray] = []
        batch_size = self._predict_batch_size()

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                stop = min(start + batch_size, len(matrix))
                batch = torch.as_tensor(matrix[start:stop], dtype=torch.float32, device=torch_device)
                logits = self.model(batch)
                logits_chunks.append(logits.detach().cpu().numpy())
        return np.concatenate(logits_chunks, axis=0)

    def _predict_scores(self, X) -> np.ndarray:
        logits = self._predict_logits(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def _build_model(self, input_dim: int):
        if self.model_name in {"logistic_regression", "svm"}:
            return _LinearBinaryModel(input_dim)
        if self.model_name == "mlp":
            hidden_layer_sizes = self.hyperparameters.get("hidden_layer_sizes", (64, 32))
            return _MLPBinaryModel(input_dim, hidden_layer_sizes=tuple(hidden_layer_sizes))
        if self.model_name == "ft_transformer":
            from isic2024_multimodal.models.tabular.ft_transformer import FTTransformerBinaryModel

            return FTTransformerBinaryModel(
                input_dim,
                d_token=int(self.hyperparameters.get("d_token", 64)),
                n_blocks=int(self.hyperparameters.get("n_blocks", 2)),
                n_heads=int(self.hyperparameters.get("n_heads", 4)),
                attention_dropout=float(self.hyperparameters.get("attention_dropout", 0.1)),
                ffn_dropout=float(self.hyperparameters.get("ffn_dropout", 0.1)),
            )
        if self.model_name == "ft_transformer_external":
            from isic2024_multimodal.models.tabular.ft_transformer import build_external_ft_transformer_binary_model

            return build_external_ft_transformer_binary_model(input_dim, self.hyperparameters)
        raise ValueError(f"Unsupported torch tabular model: {self.model_name}")

    def _max_epochs(self) -> int:
        max_iter = int(self.hyperparameters.get("max_iter", 100))
        return max(20, max_iter)

    def _patience(self) -> int:
        return min(50, max(10, self._max_epochs() // 10))

    def _predict_batch_size(self) -> int:
        default_batch_size = 2048 if self.model_name in {"ft_transformer", "ft_transformer_external"} else 65536
        return max(1, int(self.hyperparameters.get("predict_batch_size", default_batch_size)))

    def _train_batch_size(self) -> int:
        default_batch_size = 2048 if self.model_name in {"ft_transformer", "ft_transformer_external"} else 65536
        return max(1, int(self.hyperparameters.get("batch_size", default_batch_size)))

    def runtime_params(self) -> dict[str, int | str]:
        return {
            "torch_model_name": self.model_name,
            "train_batch_size": self._train_batch_size(),
            "predict_batch_size": self._predict_batch_size(),
        }

    def _learning_rate(self) -> float:
        if self.model_name in {"logistic_regression", "svm"}:
            return 5e-2
        if self.model_name == "mlp":
            return 1e-3
        if self.model_name in {"ft_transformer", "ft_transformer_external"}:
            return float(self.hyperparameters.get("learning_rate", 1e-3))
        return 1e-3

    def _weight_decay(self, num_rows: int) -> float:
        if self.model_name == "logistic_regression":
            c_value = max(float(self.hyperparameters.get("C", 1.0)), 1e-6)
            return 1.0 / (c_value * max(num_rows, 1))
        if self.model_name == "mlp":
            return float(self.hyperparameters.get("alpha", 1e-4))
        if self.model_name in {"ft_transformer", "ft_transformer_external"}:
            return float(self.hyperparameters.get("weight_decay", 1e-5))
        return 0.0

    def _svm_c_scale(self) -> float:
        return max(float(self.hyperparameters.get("C", 1.0)), 1e-6)
