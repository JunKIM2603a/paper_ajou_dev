from __future__ import annotations

import os
import sys
from pathlib import Path


EXPECTED_CONDA_ENV = "paper_ajou_dev"


def ensure_expected_conda_env(expected_env: str = EXPECTED_CONDA_ENV) -> None:
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    current_prefix = os.environ.get("CONDA_PREFIX", "")
    executable = Path(sys.executable).as_posix()

    if current_env == expected_env:
        return
    if expected_env and expected_env in current_prefix:
        return
    if f"/envs/{expected_env}/" in executable or executable.endswith(f"/envs/{expected_env}/bin/python"):
        return

    raise RuntimeError(
        f"This benchmark must run inside the conda environment '{expected_env}'. "
        f"Current executable: {sys.executable}"
    )
