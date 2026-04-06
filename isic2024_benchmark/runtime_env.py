from __future__ import annotations

import os
import sys
from pathlib import Path


EXPECTED_CONDA_ENV = "paper_ajou_dev"
EXPECTED_CONDA_ENV_OVERRIDE_VAR = "ISIC2024_EXPECTED_CONDA_ENV"
DEFAULT_MLFLOW_FILE_TRACKING_URI = "file:./mlruns"
DEFAULT_MLFLOW_SQLITE_TRACKING_URI = "sqlite:///mlflow.db"


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
