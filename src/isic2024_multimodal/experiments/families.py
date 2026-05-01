from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_FAMILIES = {
    "tabular_baselines": "ISIC2024-Tabular-Baselines",
    "image_baselines": "ISIC2024-Image-Baselines",
    "multimodal_baselines": "ISIC2024-Multimodal-Baselines",
    "final_paper_model": "ISIC2024-Final-Paper-Model",
}


@dataclass(frozen=True)
class FamilyPaths:
    family: str
    run_group_id: str
    output_root: Path
    table_root: Path
    run_manifest_path: Path
    preflight_path: Path
    status_path: Path


def make_run_group_id(family: str) -> str:
    return f"{family}_{time.strftime('%Y%m%d_%H%M%S')}"


def load_suite_config(path: str | Path, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root or Path.cwd()).resolve()
    suite_path = resolve_path(path, repo_root=repo_root)
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    payload["_suite_path"] = str(suite_path)
    return payload


def resolve_family_paths(
    *,
    family: str,
    run_group_id: str,
    repo_root: str | Path | None = None,
    smoke: bool = False,
) -> FamilyPaths:
    if family not in EXPERIMENT_FAMILIES:
        raise ValueError(f"Unknown experiment family: {family}")
    if not run_group_id or "/" in run_group_id or "\\" in run_group_id:
        raise ValueError(f"Unsafe run_group_id: {run_group_id!r}")
    repo_root = Path(repo_root or Path.cwd()).resolve()
    if smoke:
        output_root = repo_root / "experiments" / "outputs" / "smoke" / run_group_id / family
        table_root = repo_root / "experiments" / "tables" / "smoke" / run_group_id / family
    else:
        output_root = repo_root / "experiments" / "outputs" / family / run_group_id
        table_root = repo_root / "experiments" / "tables" / family / run_group_id
    return FamilyPaths(
        family=family,
        run_group_id=run_group_id,
        output_root=output_root,
        table_root=table_root,
        run_manifest_path=output_root / "run_manifest.json",
        preflight_path=output_root / "preflight.json",
        status_path=output_root / "family_status.json",
    )


def reset_family_outputs(paths: FamilyPaths, *, dry_run: bool = False) -> list[Path]:
    targets = [paths.output_root, paths.table_root]
    if dry_run:
        return targets
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
    return targets


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_root / value

