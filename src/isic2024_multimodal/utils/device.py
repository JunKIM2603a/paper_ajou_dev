from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceResolution:
    requested_device: str
    resolved_device: str
    cuda_available: bool
    visible_device_count: int
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DeviceListResolution:
    requested_devices: list[int] | None
    resolved_devices: list[int]
    device_policy: str
    cuda_available: bool
    visible_device_count: int
    fallback_reason: str | None = None

    @property
    def uses_cuda(self) -> bool:
        return bool(self.resolved_devices)


def resolve_device(requested_device: str | None = "auto") -> DeviceResolution:
    requested = (requested_device or "auto").strip().lower()
    if requested == "gpu":
        requested = "cuda"
    if requested in {"auto", "cuda"} or requested.startswith("cuda:"):
        status = _cuda_status()
    else:
        status = _cuda_status()
        return DeviceResolution(
            requested_device=requested,
            resolved_device="cpu",
            cuda_available=bool(status["available"]),
            visible_device_count=int(status["count"]),
            fallback_reason=None,
        )

    if requested == "auto":
        if status["usable"]:
            return DeviceResolution(
                requested_device=requested,
                resolved_device="cuda",
                cuda_available=True,
                visible_device_count=status["count"],
                fallback_reason=None,
            )
        return DeviceResolution(
            requested_device=requested,
            resolved_device="cpu",
            cuda_available=bool(status["available"]),
            visible_device_count=status["count"],
            fallback_reason=status["reason"],
        )

    if not status["usable"]:
        return DeviceResolution(
            requested_device=requested,
            resolved_device="cpu",
            cuda_available=bool(status["available"]),
            visible_device_count=status["count"],
            fallback_reason=status["reason"],
        )

    requested_status = _can_allocate(requested)
    if requested_status["usable"]:
        return DeviceResolution(
            requested_device=requested,
            resolved_device=requested,
            cuda_available=True,
            visible_device_count=status["count"],
            fallback_reason=None,
        )

    return DeviceResolution(
        requested_device=requested,
        resolved_device="cpu",
        cuda_available=bool(status["available"]),
        visible_device_count=status["count"],
        fallback_reason=requested_status["reason"] or status["reason"],
    )


def resolve_device_list(
    requested_devices: list[int] | None,
    *,
    device_policy: str = "auto",
) -> DeviceListResolution:
    policy = (device_policy or "auto").strip().lower()
    normalized_requested = list(requested_devices) if requested_devices else None
    if policy == "cpu":
        status = _cuda_status()
        return DeviceListResolution(
            requested_devices=normalized_requested,
            resolved_devices=[],
            device_policy=policy,
            cuda_available=bool(status["available"]),
            visible_device_count=int(status["count"]),
            fallback_reason="device_policy=cpu",
        )
    if policy != "auto":
        raise ValueError(f"Unsupported device policy: {device_policy}")

    status = _cuda_status()
    if not status["usable"]:
        return DeviceListResolution(
            requested_devices=normalized_requested,
            resolved_devices=[],
            device_policy=policy,
            cuda_available=bool(status["available"]),
            visible_device_count=status["count"],
            fallback_reason=status["reason"],
        )

    if normalized_requested is None:
        return DeviceListResolution(
            requested_devices=None,
            resolved_devices=[0],
            device_policy=policy,
            cuda_available=True,
            visible_device_count=status["count"],
            fallback_reason=None,
        )

    if len(set(normalized_requested)) != len(normalized_requested):
        return DeviceListResolution(
            requested_devices=normalized_requested,
            resolved_devices=[],
            device_policy=policy,
            cuda_available=True,
            visible_device_count=status["count"],
            fallback_reason=f"duplicate GPU ids requested: {normalized_requested}",
        )

    valid_devices = [
        device
        for device in normalized_requested
        if device >= 0 and device < status["count"] and _can_allocate(f"cuda:{device}")["usable"]
    ]
    if valid_devices:
        fallback_reason = None
        if len(valid_devices) != len(normalized_requested):
            fallback_reason = f"using available GPU subset {valid_devices} from requested {normalized_requested}"
        return DeviceListResolution(
            requested_devices=normalized_requested,
            resolved_devices=valid_devices,
            device_policy=policy,
            cuda_available=True,
            visible_device_count=status["count"],
            fallback_reason=fallback_reason,
        )

    return DeviceListResolution(
        requested_devices=normalized_requested,
        resolved_devices=[],
        device_policy=policy,
        cuda_available=True,
        visible_device_count=status["count"],
        fallback_reason=f"no requested GPU ids are usable: {normalized_requested}",
    )


def _cuda_status() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "count": 0, "usable": False, "reason": f"torch import failed: {exc}"}

    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
    except Exception as exc:
        return {
            "available": False,
            "count": 0,
            "usable": False,
            "reason": f"CUDA availability check failed: {exc}",
        }
    if not available:
        return {"available": False, "count": count, "usable": False, "reason": "torch.cuda.is_available() is False"}
    if count <= 0:
        return {"available": True, "count": count, "usable": False, "reason": "no visible CUDA devices"}
    allocation = _can_allocate("cuda")
    return {
        "available": True,
        "count": count,
        "usable": allocation["usable"],
        "reason": allocation["reason"],
    }


def _can_allocate(device: str) -> dict[str, object]:
    try:
        import torch

        torch.empty(1, device=device)
    except Exception as exc:
        return {"usable": False, "reason": f"failed to allocate tensor on {device}: {exc}"}
    return {"usable": True, "reason": None}
