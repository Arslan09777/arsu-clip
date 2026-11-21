"""Report structured pipeline activity without changing simple progress callbacks."""
from inspect import getattr_static
from typing import Callable


Progress = Callable[[float, str], None]


def report(progress: Progress, pct: float, message: str, *, phase: str, **detail: object) -> None:
    if getattr_static(progress, "detail", None) is not None:
        progress.detail(pct, message, {"phase": phase, **{k: v for k, v in detail.items() if v is not None}})
    else:
        progress(pct, message)
