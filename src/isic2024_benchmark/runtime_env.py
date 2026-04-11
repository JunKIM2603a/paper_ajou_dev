from __future__ import annotations

import os
import sys
from pathlib import Path


EXPECTED_CONDA_ENV = "paper_ajou_dev"
EXPECTED_CONDA_ENV_OVERRIDE_VAR = "ISIC2024_EXPECTED_CONDA_ENV"
DEFAULT_MLFLOW_FILE_TRACKING_URI = "file:./mlruns"
DEFAULT_MLFLOW_SQLITE_TRACKING_URI = "sqlite:///mlflow.db"


def load_project_env(env_path: str | Path = ".env") -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)
        loaded[key] = value

    # Hugging Face client libraries commonly look for HF_TOKEN.
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    return loaded


def ensure_expected_conda_env(expected_env: str = EXPECTED_CONDA_ENV) -> None:
    override_env = os.environ.get(EXPECTED_CONDA_ENV_OVERRIDE_VAR, "").strip()
    resolved_expected_env = override_env or expected_env
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    current_prefix = os.environ.get("CONDA_PREFIX", "")
    executable = Path(sys.executable).as_posix()

    if current_env == resolved_expected_env:
        return
    if resolved_expected_env and resolved_expected_env in current_prefix:
        return
    if f"/envs/{resolved_expected_env}/" in executable or executable.endswith(
        f"/envs/{resolved_expected_env}/bin/python"
    ):
        return

    raise RuntimeError(
        f"This benchmark must run inside the conda environment '{resolved_expected_env}'. "
        f"Current executable: {sys.executable}"
    )


def get_default_mlflow_tracking_uri() -> str:
    explicit_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit_uri:
        return explicit_uri
    if Path("mlflow.db").exists():
        return DEFAULT_MLFLOW_SQLITE_TRACKING_URI
    return DEFAULT_MLFLOW_FILE_TRACKING_URI
