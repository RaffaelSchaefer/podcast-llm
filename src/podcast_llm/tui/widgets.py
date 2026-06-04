from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..progress import PHASE_LABELS

WIZARD_STEP_NAMES: tuple[str, ...] = ("Sources", "Style", "Voices & Engine", "Review")
GENERATION_PHASES: tuple[str, ...] = ("parse", "outline", "script", "synthesize", "export")


class _RailStep(Static):
    """A single clickable row in the wizard step rail."""

    def __init__(self, index: int, *, active_index: int, max_index: int, **kwargs) -> None:
        self._index = index
        name = WIZARD_STEP_NAMES[index]
        number = index + 1
        if index == active_index:
            state = "current"
            marker = "▸"
        elif index <= max_index:
            state = "visited"
            marker = "✓" if index < active_index else "•"
        else:
            state = "future"
            marker = "○"
        self._clickable = state == "visited"
        super().__init__(f"{marker}  {number}. {name}", **kwargs)
        self.add_class(f"rail-{state}")

    def on_click(self) -> None:
        if self._clickable:
            self.app.goto_step(self._index)


class StepRail(Vertical):
    """Persistent left rail listing the wizard steps and their state.

    Visited/completed steps are clickable to jump back to them; the current
    step is highlighted and future steps are dim and inert."""

    def __init__(self, active_index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_index = active_index

    def compose(self) -> ComposeResult:
        max_index = getattr(self.app, "_max_index", self._active_index)
        yield Static("Steps", classes="rail-title")
        for index in range(len(WIZARD_STEP_NAMES)):
            yield _RailStep(index, active_index=self._active_index, max_index=max_index)


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
