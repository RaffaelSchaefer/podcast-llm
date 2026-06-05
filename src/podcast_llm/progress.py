from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Phase = Literal["parse", "outline", "script", "synthesize", "export", "done"]
ProgressKind = Literal["status", "text_reset", "text_fragment", "text_replace"]
TextScope = Literal["outline", "script"]

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
    seg: int = 0  # segment index (1-based) during synthesize, 0 = unset
    round_current: int = 0  # turn index within the current segment
    round_total: int = 0  # total turns in the current segment
    speaker: str | None = None
    delivery_instruction: str = ""
    kind: ProgressKind = "status"
    text: str = ""
    text_scope: TextScope | None = None
    segment_title: str = ""


ProgressCallback = Callable[[ProgressEvent], None]
