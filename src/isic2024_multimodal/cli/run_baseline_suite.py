from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from isic2024_multimodal.cli.run_experiment_family import build_subprocess_env
from isic2024_multimodal.experiments.families import EXPERIMENT_FAMILIES
from isic2024_multimodal.utils.progress import format_eta, format_progress_duration, progress_index_label
from isic2024_multimodal.utils.runtime_env import ensure_expected_conda_env, load_project_env


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

DEFAULT_BASELINE_FAMILIES = ("tabular_baselines", "image_baselines")
SUPPORTED_BASELINE_FAMILIES = ("tabular_baselines", "image_baselines", "multimodal_baselines")


@dataclass(frozen=True)
class FamilyCommand:
    family: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the configured ISIC2024 baseline experiment families with one command. "
            "By default this runs implemented tabular and image baseline suites."
        )
    )
    parser.add_argument(
        "--families",
        nargs="*",
        choices=SUPPORTED_BASELINE_FAMILIES,
        default=list(DEFAULT_BASELINE_FAMILIES),
        help="Baseline families to run. multimodal_baselines is opt-in because the runner is still a scaffold.",
    )
    parser.add_argument("--run-group-id", default=None, help="Shared run group id. Defaults to baseline_suite_<timestamp>.")
    parser.add_argument("--devices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--device-policy",
        choices=["auto", "cpu"],
        default="auto",
        help="Passed through to each family runner.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use smoke caps from each suite config.")
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Image-only batch size override passed through to image_baselines.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Write each family preflight summary only.")
    parser.add_argument("--resume", action="store_true", help="Skip a family if its previous status is ok.")
    parser.add_argument("--reset-family-output", action="store_true", help="Reset each selected family output/table root.")
    parser.add_argument("--skip-reports", action="store_true", help="Skip per-family report generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned family commands without running them.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue later families even if an earlier family fails.",
    )
    return parser.parse_args()


def main() -> None:
    load_project_env()
    ensure_expected_conda_env()
    args = parse_args()
    args.run_group_id = args.run_group_id or make_run_group_id()
    family_commands = build_suite_commands(args)
    print(
        json.dumps(
            {
                "run_group_id": args.run_group_id,
                "families": [entry.family for entry in family_commands],
                "dry_run": bool(args.dry_run),
                "commands": [entry.command for entry in family_commands],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return

    results = run_suite_commands(family_commands, continue_on_failure=args.continue_on_failure)
    print(json.dumps({"run_group_id": args.run_group_id, "results": results}, ensure_ascii=False, indent=2))
    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        raise SystemExit(int(failures[0]["returncode"]))


def build_suite_commands(args: argparse.Namespace) -> list[FamilyCommand]:
    families = list(args.families or DEFAULT_BASELINE_FAMILIES)
    unsupported = sorted(set(families) - set(SUPPORTED_BASELINE_FAMILIES))
    if unsupported:
        raise ValueError(f"Unsupported baseline families: {unsupported}")
    return [FamilyCommand(family=family, command=build_family_command(family, args)) for family in families]


def build_family_command(family: str, args: argparse.Namespace) -> list[str]:
    if family not in EXPERIMENT_FAMILIES:
        raise ValueError(f"Unknown experiment family: {family}")
    command = [
        sys.executable,
        "-m",
        "isic2024_multimodal.cli.run_experiment_family",
        "--family",
        family,
        "--run-group-id",
        args.run_group_id,
        "--device-policy",
        args.device_policy,
    ]
    if args.devices:
        command.append("--devices")
        command.extend(str(device) for device in args.devices)
    batch_size_override = getattr(args, "batch_size_override", None)
    if family == "image_baselines" and batch_size_override is not None:
        command.extend(["--batch-size-override", str(batch_size_override)])
    for attr, flag in (
        ("smoke", "--smoke"),
        ("preflight_only", "--preflight-only"),
        ("resume", "--resume"),
        ("reset_family_output", "--reset-family-output"),
        ("skip_reports", "--skip-reports"),
    ):
        if getattr(args, attr, False):
            command.append(flag)
    return command


def run_suite_commands(
    family_commands: list[FamilyCommand],
    *,
    continue_on_failure: bool,
) -> list[dict[str, str | int | float]]:
    env = build_subprocess_env()
    results: list[dict[str, str | int | float]] = []
    suite_start = time.time()
    total_families = len(family_commands)
    for family_index, entry in enumerate(family_commands, start=1):
        started = time.time()
        print(
            f"[run_baseline_suite] Start family={progress_index_label(family_index, total_families)} "
            f"{entry.family} completed_families={len(results)} "
            f"elapsed={format_progress_duration(time.time() - suite_start)} "
            f"eta={format_eta(elapsed_seconds=time.time() - suite_start, completed_count=len(results), total_count=total_families)}",
            flush=True,
        )
        result = subprocess.run(entry.command, cwd=REPO_ROOT, env=env, check=False)
        elapsed = time.time() - started
        results.append(
            {
                "family": entry.family,
                "returncode": int(result.returncode),
                "duration_seconds": elapsed,
            }
        )
        print(
            f"[run_baseline_suite] Finished family={progress_index_label(family_index, total_families)} "
            f"{entry.family} returncode={result.returncode} "
            f"duration={format_progress_duration(elapsed)} completed_families={len(results)}/{total_families} "
            f"elapsed={format_progress_duration(time.time() - suite_start)} "
            f"eta={format_eta(elapsed_seconds=time.time() - suite_start, completed_count=len(results), total_count=total_families)}",
            flush=True,
        )
        if result.returncode != 0 and not continue_on_failure:
            break
    return results


def make_run_group_id() -> str:
    return f"baseline_suite_{time.strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    main()
