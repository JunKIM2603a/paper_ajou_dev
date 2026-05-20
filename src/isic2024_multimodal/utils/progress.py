from __future__ import annotations


def format_progress_duration(seconds: float | int | None) -> str:
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


def estimate_remaining_seconds(
    elapsed_seconds: float | int,
    completed_count: int,
    total_count: int,
) -> float | None:
    if total_count <= 0:
        return 0.0
    remaining_count = max(int(total_count) - int(completed_count), 0)
    if remaining_count == 0:
        return 0.0
    if completed_count <= 0:
        return None
    seconds_per_item = float(elapsed_seconds) / max(int(completed_count), 1)
    return seconds_per_item * remaining_count


def progress_index_label(index: int, total: int) -> str:
    return f"{int(index)}/{int(total)}"


def format_eta(
    *,
    elapsed_seconds: float | int,
    completed_count: int,
    total_count: int,
) -> str:
    return format_progress_duration(
        estimate_remaining_seconds(
            elapsed_seconds=elapsed_seconds,
            completed_count=completed_count,
            total_count=total_count,
        )
    )
