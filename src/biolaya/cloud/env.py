from __future__ import annotations

import platform


def resolve_device(requested: str | None = "auto") -> str:
    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "gpu":
        requested = "cuda"

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to resolve a BioLaya device.") from exc

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested, but PyTorch cannot see a CUDA GPU. "
                "Use --device cpu for a local CPU test or enable a GPU runtime in Colab."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def environment_report(requested_device: str = "auto") -> dict:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": requested_device,
    }
    try:
        import torch
        report.update(
            torch=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            bf16_supported=bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        )
        try:
            report["resolved_device"] = resolve_device(requested_device)
        except Exception as exc:
            report["device_error"] = str(exc)
    except Exception as exc:
        report["torch_error"] = repr(exc)
    try:
        import laya
        report["laya"] = getattr(laya, "__version__", "unknown")
    except Exception as exc:
        report["laya_error"] = repr(exc)
    return report
