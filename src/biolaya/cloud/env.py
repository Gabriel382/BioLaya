from __future__ import annotations

import platform


def environment_report() -> dict:
    report = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        report.update(
            torch=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            bf16_supported=bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        )
    except Exception as exc:
        report["torch_error"] = repr(exc)
    try:
        import laya
        report["laya"] = getattr(laya, "__version__", "unknown")
    except Exception as exc:
        report["laya_error"] = repr(exc)
    return report
