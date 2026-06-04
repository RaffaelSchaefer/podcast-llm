from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from ..widgets import StepRail


class WizardScreen(Screen):
    """Shared full-bleed frame for a single wizard step: a persistent left step
    rail, a titled content column with the step body, a status line, and a
    Back/Next nav row."""

    STEP_INDEX: int = 0
    CARD_TITLE: str = ""
    NEXT_LABEL: str = "Next"

    BINDINGS = [
        Binding("ctrl+n", "next", "Next", priority=True),
        Binding("ctrl+b", "back", "Back", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(classes="wizard-frame"):
            yield StepRail(self.STEP_INDEX, classes="step-rail")
            with Vertical(classes="wizard-content"):
                yield Static(self.CARD_TITLE, classes="content-title")
                with VerticalScroll(classes="card-body"):
                    yield from self.compose_body()
                yield Static("", classes="wizard-status")
                with Horizontal(classes="nav-row"):
                    yield Button("◀  Back", id="nav_back", disabled=self.STEP_INDEX == 0)
                    yield Button(f"{self.NEXT_LABEL}  ▶", id="nav_next", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.populate()

    # --- hooks for subclasses ---
    def compose_body(self) -> ComposeResult:
        return []

    def populate(self) -> None:
        """Fill widgets from ``self.app.draft``."""

    def commit(self, strict: bool) -> None:
        """Write widgets into ``self.app.draft``. Raise ValueError when
        ``strict`` and the step is incomplete."""

    # --- navigation ---
    def action_next(self) -> None:
        try:
            self.commit(strict=True)
        except ValueError as exc:
            self.set_status(str(exc))
            return
        self.app.advance()

    def action_back(self) -> None:
        if self.STEP_INDEX == 0:
            return
        try:
            self.commit(strict=False)
        except ValueError:
            pass
        self.app.go_back()

    @on(Button.Pressed, "#nav_next")
    def _on_next(self, event: Button.Pressed) -> None:
        self.action_next()

    @on(Button.Pressed, "#nav_back")
    def _on_back(self, event: Button.Pressed) -> None:
        self.action_back()

    def set_status(self, text: str) -> None:
        self.query_one(".wizard-status", Static).update(text)
