from __future__ import annotations

try:
    from .builder import build_dataloader, build_dataloaders
except ImportError as exc:  # pragma: no cover - only used when mindspore is unavailable.
    if exc.name and exc.name.startswith("mindspore"):
        def _missing_mindspore(*args, **kwargs):
            raise ImportError(
                "mindspore is required to build dataloaders. Install mindspore before "
                "importing build_dataloader or build_dataloaders."
            ) from exc

        build_dataloader = _missing_mindspore
        build_dataloaders = _missing_mindspore
    else:
        raise

__all__ = ["build_dataloader", "build_dataloaders"]
