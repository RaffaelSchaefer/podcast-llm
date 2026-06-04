import os
import subprocess
import sys
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ProgressBar, RichLog, Static

from ...models import GenerationRequest, GenerationResult
from ...pipeline import PodcastPipeline
from ...progress import PHASE_LABELS, ProgressEvent
from ...tts import preflight_qwen_runtime
from ..widgets import PhaseChecklist


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

    def __init__(self, request: GenerationRequest, *, auto_start: bool = True) -> None:
        super().__init__()
        self._request = request
        self._auto_start = auto_start
        self._output_dir: Path | None = None
        self._seg_count: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(classes="gen-content"):
            yield Static("Generating your podcast…", id="gen_title")
            yield PhaseChecklist(id="phases")
            yield ProgressBar(id="gen_bar", show_eta=False)
            yield Static("", id="round_label")
            yield ProgressBar(id="round_bar", show_eta=False)
            yield RichLog(id="gen_log", markup=True, wrap=True)
            with Horizontal(id="gen_nav"):
                yield Button("Open folder", id="gen_open", variant="primary", disabled=True)
                yield Button("Restart", id="gen_restart", disabled=True)
                yield Button("Quit", id="gen_quit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#gen_bar", ProgressBar).update(total=None)
        self.query_one("#round_bar", ProgressBar).update(total=None)
        if self._auto_start:
            try:
                preflight_qwen_runtime()
            except RuntimeError as exc:
                self.post_message(GenerationFailed(exc))
                return
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

        bar = self.query_one("#gen_bar", ProgressBar)
        round_bar = self.query_one("#round_bar", ProgressBar)
        round_label = self.query_one("#round_label", Static)

        if event.phase == "synthesize" and event.total > 0:
            bar.update(total=event.total, progress=event.current)
            if event.round_total > 0:
                round_bar.update(total=event.round_total, progress=event.round_current)
                seg_display = f"{event.seg} / {self._seg_count}" if self._seg_count else str(event.seg)
                round_label.update(f"[dim]Segment {seg_display}[/dim]")
        else:
            if event.phase == "script" and event.total > 0:
                self._seg_count = event.total
            bar.update(total=None)
            round_bar.update(total=None)
            round_label.update("")

        if event.message:
            self.query_one("#gen_log", RichLog).write(f"[dim]{label}[/dim]  {event.message}")

    def on_generation_done(self, message: GenerationDone) -> None:
        result = message.result
        self._output_dir = Path(result.output_dir)
        self.query_one("#phases", PhaseChecklist).set_phase("done")

        bar = self.query_one("#gen_bar", ProgressBar)
        if bar.total:
            bar.update(progress=bar.total)
        round_bar = self.query_one("#round_bar", ProgressBar)
        if round_bar.total:
            round_bar.update(progress=round_bar.total)

        log = self.query_one("#gen_log", RichLog)
        log.write(f"[green b]Done.[/green b] Saved to {result.output_dir}")
        log.write(f"Transcript: {result.transcript_path}")
        log.write(f"Audio: {result.audio_path}")
        self.query_one("#gen_title", Static).update(
            f"[green]Podcast ready[/green] — {result.output_dir}"
        )
        self.query_one("#gen_open", Button).disabled = False
        self.query_one("#gen_restart", Button).disabled = False

    def on_generation_failed(self, message: GenerationFailed) -> None:
        self.query_one("#gen_log", RichLog).write(f"[red b]Error:[/red b] {message.error}")
        self.query_one("#gen_title", Static).update(f"[red]Generation failed[/red] - {message.error}")
        self.query_one("#gen_restart", Button).disabled = False

    @on(Button.Pressed, "#gen_open")
    def _on_open(self, event: Button.Pressed) -> None:
        if self._output_dir is None:
            return
        try:
            self._open_in_file_manager(self._output_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI log
            self.query_one("#gen_log", RichLog).write(f"[red]Could not open folder:[/red] {exc}")

    @staticmethod
    def _open_in_file_manager(path: Path) -> None:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    @on(Button.Pressed, "#gen_restart")
    def _on_restart(self, event: Button.Pressed) -> None:
        self.app.pop_screen()
        self.app.restart()

    @on(Button.Pressed, "#gen_quit")
    def _on_quit(self, event: Button.Pressed) -> None:
        self.app.exit()
