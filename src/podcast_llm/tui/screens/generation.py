from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ProgressBar, RichLog, Static

from ...models import GenerationRequest, GenerationResult
from ...pipeline import PodcastPipeline
from ...progress import PHASE_LABELS, ProgressEvent
from ..widgets import PhaseAnimation, PhaseChecklist


class ProgressUpdate(Message):
    def __init__(self, event: ProgressEvent) -> None:
        self.event = event
        super().__init__()


class GenerationDone(Message):
    def __init__(self, result: GenerationResult) -> None:
        self.result = result
        super().__init__()


class GenerationFailed(Message):
    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__()


class GenerationScreen(Screen):
    """Runs the pipeline in a worker thread and streams live progress."""

    BINDINGS = [Binding("l", "toggle_log", "Log view", priority=True)]

    def __init__(self, request: GenerationRequest, *, auto_start: bool = True) -> None:
        super().__init__()
        self._request = request
        self._auto_start = auto_start
        self._show_log = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(classes="gen-card"):
            yield Static("Generating your podcast...", id="gen_title")
            yield Static("", id="view_hint")
            with Vertical(id="theater_panel"):
                yield PhaseAnimation(id="phase_anim")
                yield Static("Waiting for the first generation stage...", id="phase_caption")
            yield ProgressBar(id="gen_bar", show_eta=False)
            with Vertical(id="log_panel"):
                yield PhaseChecklist(id="phases")
                yield RichLog(id="gen_log", markup=True, wrap=True)
            with Horizontal(id="gen_nav"):
                yield Button("<  Back to review", id="gen_back", disabled=True)
                yield Button("Quit", id="gen_quit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(".gen-card", Vertical).border_title = "Generate"
        self.query_one("#gen_bar", ProgressBar).update(total=None)
        self._sync_view_mode()
        if self._auto_start:
            self._run()

    @work(thread=True, exclusive=True)
    def _run(self) -> None:
        def callback(event: ProgressEvent) -> None:
            self.post_message(ProgressUpdate(event))

        try:
            result = PodcastPipeline().generate(self._request, callback)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI log
            self.post_message(GenerationFailed(exc))
            return
        self.post_message(GenerationDone(result))

    def on_progress_update(self, message: ProgressUpdate) -> None:
        event = message.event
        label = PHASE_LABELS.get(event.phase, event.phase)

        self.query_one("#phases", PhaseChecklist).set_phase(event.phase)
        self.query_one("#phase_anim", PhaseAnimation).set_phase(event.phase, speaker=event.speaker)

        bar = self.query_one("#gen_bar", ProgressBar)
        if event.phase == "synthesize" and event.total > 0:
            bar.update(total=event.total, progress=event.current)

        caption = label if not event.message else f"{label}: {event.message}"
        self.query_one("#phase_caption", Static).update(caption)
        if event.message:
            self.query_one("#gen_log", RichLog).write(f"[dim]{label}[/dim]  {event.message}")

    def on_generation_done(self, message: GenerationDone) -> None:
        result = message.result
        self.query_one("#phases", PhaseChecklist).set_phase("done")
        self.query_one("#phase_anim", PhaseAnimation).set_phase("done")
        self.query_one("#phase_caption", Static).update(f"Done: saved to {result.output_dir}")

        bar = self.query_one("#gen_bar", ProgressBar)
        if bar.total:
            bar.update(progress=bar.total)

        log = self.query_one("#gen_log", RichLog)
        log.write(f"[green b]Done.[/green b] Saved to {result.output_dir}")
        log.write(f"Transcript: {result.transcript_path}")
        log.write(f"Audio: {result.audio_path}")
        self.query_one("#gen_title", Static).update("[green]Podcast ready[/green]")
        self.query_one("#gen_back", Button).disabled = False

    def on_generation_failed(self, message: GenerationFailed) -> None:
        self.query_one("#phase_anim", PhaseAnimation).set_phase("failed")
        self.query_one("#phase_caption", Static).update("Generation stopped. Press L for the error log.")
        self.query_one("#gen_log", RichLog).write(f"[red b]Error:[/red b] {message.error}")
        self.query_one("#gen_title", Static).update("[red]Generation failed[/red]")
        self.query_one("#gen_back", Button).disabled = False

    def action_toggle_log(self) -> None:
        self._show_log = not self._show_log
        self._sync_view_mode()

    def _sync_view_mode(self) -> None:
        self.query_one("#theater_panel", Vertical).display = not self._show_log
        self.query_one("#log_panel", Vertical).display = self._show_log
        if self._show_log:
            hint = "Log view - press L for theater mode"
        else:
            hint = "Theater mode - press L for log view"
        self.query_one("#view_hint", Static).update(f"[dim]{hint}[/dim]")

    @on(Button.Pressed, "#gen_back")
    def _on_back(self, event: Button.Pressed) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#gen_quit")
    def _on_quit(self, event: Button.Pressed) -> None:
        self.app.exit()
