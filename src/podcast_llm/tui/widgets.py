from textual.widgets import Static

from ..progress import PHASE_LABELS
from .ascii_art import FRAMES

WIZARD_STEP_NAMES: tuple[str, ...] = ("Sources", "Style", "Voices & Engine", "Review")
GENERATION_PHASES: tuple[str, ...] = ("parse", "outline", "script", "synthesize", "export")


class StepIndicator(Static):
    """Top-of-screen breadcrumb showing wizard progress."""

    def __init__(self, active_index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_index = active_index

    def on_mount(self) -> None:
        chunks: list[str] = []
        for index, name in enumerate(WIZARD_STEP_NAMES):
            number = index + 1
            if index < self._active_index:
                chunks.append(f"[green]✓ {name}[/green]")
            elif index == self._active_index:
                chunks.append(f"[b reverse] {number}  {name} [/]")
            else:
                chunks.append(f"[dim]{number}  {name}[/dim]")
        self.update("  [dim]→[/dim]  ".join(chunks))


class PhaseChecklist(Static):
    """Live checklist of generation phases on the progress screen."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_index = -1

    def on_mount(self) -> None:
        self._render_lines()

    def set_phase(self, phase: str) -> None:
        if phase == "done":
            self._max_index = len(GENERATION_PHASES)
        elif phase in GENERATION_PHASES:
            self._max_index = max(self._max_index, GENERATION_PHASES.index(phase))
        self._render_lines()

    def _render_lines(self) -> None:
        lines: list[str] = []
        for index, phase in enumerate(GENERATION_PHASES):
            label = PHASE_LABELS[phase]
            if index < self._max_index:
                lines.append(f"[green]✓[/green]  {label}")
            elif index == self._max_index:
                lines.append(f"[yellow]▶[/yellow]  [b]{label}[/b]")
            else:
                lines.append(f"[dim]○  {label}[/dim]")
        self.update("\n".join(lines))


class PhaseAnimation(Static):
    """Looping ASCII-art panel whose animation switches with the active phase.

    Frames for each phase live in ``ascii_art.FRAMES``; a repeating timer cycles
    through the current phase's frames so each pipeline stage reads as a small
    animation."""

    FRAME_INTERVAL = 0.22

    def __init__(self, phase: str = "parse", speaker: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._phase = phase
        self._speaker = self._speaker_key(speaker)
        self._frame_key = self._select_frame_key(phase, self._speaker)
        self._frames = FRAMES[self._frame_key]
        self._index = 0

    def on_mount(self) -> None:
        self._render_frame()
        self.set_interval(self.FRAME_INTERVAL, self._tick)

    def set_phase(self, phase: str, speaker: str | None = None) -> None:
        """Switch to ``phase``'s animation, restarting it from the first frame.

        Unknown phases and repeats of the current phase/speaker combination are
        no-ops so an in-flight loop keeps running smoothly across repeated
        progress events."""
        speaker_key = self._speaker_key(speaker)
        frame_key = self._select_frame_key(phase, speaker_key)
        if frame_key is None:
            return
        if frame_key == self._frame_key:
            return
        self._phase = phase
        self._speaker = speaker_key
        self._frame_key = frame_key
        self._frames = FRAMES[frame_key]
        self._index = 0
        self._render_frame()

    def _tick(self) -> None:
        self._index = (self._index + 1) % len(self._frames)
        self._render_frame()

    def _render_frame(self) -> None:
        self.update(self._frames[self._index])

    @staticmethod
    def _speaker_key(speaker: str | None) -> str | None:
        if speaker is None:
            return None
        normalized = speaker.strip().lower()
        if normalized in {"a", "host a", "host_a"}:
            return "A"
        if normalized in {"b", "host b", "host_b"}:
            return "B"
        return None

    @staticmethod
    def _select_frame_key(phase: str, speaker: str | None) -> str | None:
        if phase == "synthesize" and speaker is not None:
            speaker_key = f"synthesize:{speaker}"
            if speaker_key in FRAMES:
                return speaker_key
        if phase in FRAMES:
            return phase
        return None
