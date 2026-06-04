from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Phase = Literal["parse", "outline", "script", "synthesize", "export", "done"]

PHASE_ORDER: tuple[Phase, ...] = ("parse", "outline", "script", "synthesize", "export", "done")

PHASE_LABELS: dict[Phase, str] = {
    "parse": "Parse sources",
    "outline": "Design outline",
    "script": "Write script",
    "synthesize": "Synthesize audio",
    "export": "Export episode",
    "done": "Done",
}


@dataclass(frozen=True)
class ProgressEvent:
    phase: Phase
    message: str = ""
    current: int = 0  # 0 means indeterminate
    total: int = 0  # 0 until the total is known
    speaker: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]
