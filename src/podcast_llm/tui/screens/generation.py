import json
import os
import re
import subprocess
import sys
from pathlib import Path

from rich.markup import escape
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
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
        self._live_buffer = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(classes="gen-content"):
            yield Static("Generating your podcast…", id="gen_title")
            yield PhaseChecklist(id="phases")
            yield ProgressBar(id="gen_bar", show_eta=False)
            yield Static("", id="round_label")
            yield ProgressBar(id="round_bar", show_eta=False)
            with VerticalScroll(id="live_panel"):
                yield Static("", id="live_text", markup=True)
            yield RichLog(id="gen_log", markup=True, wrap=True)
            with Horizontal(id="gen_nav"):
                yield Button("Open folder", id="gen_open", variant="primary", disabled=True)
                yield Button("Restart", id="gen_restart", disabled=True)
                yield Button("Quit", id="gen_quit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#gen_bar", ProgressBar).display = False
        self.query_one("#round_bar", ProgressBar).display = False
        self.query_one("#round_label", Static).display = False
        panel = self.query_one("#live_panel", VerticalScroll)
        panel.border_title = "Live text"
        self.query_one("#live_text", Static).update("[dim]LLM output will appear here…[/dim]")
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
        if event.kind != "status":
            self._update_live_text(event)
            return

        bar = self.query_one("#gen_bar", ProgressBar)
        round_bar = self.query_one("#round_bar", ProgressBar)
        round_label = self.query_one("#round_label", Static)

        if event.phase == "synthesize" and event.total > 0:
            bar.display = True
            bar.update(total=event.total, progress=event.current)
            if event.round_total > 0:
                round_bar.display = True
                round_label.display = True
                round_bar.update(total=event.round_total, progress=event.round_current)
                seg_display = f"{event.seg} / {self._seg_count}" if self._seg_count else str(event.seg)
                round_label.update(f"[dim]Segment {seg_display}[/dim]")
            else:
                round_bar.display = False
                round_label.display = False
            if event.speaker:
                self._show_synthesis_turn(event)
        else:
            if event.phase == "script" and event.total > 0:
                self._seg_count = event.total
            bar.display = False
            round_bar.display = False
            round_label.display = False
            bar.update(total=None, progress=0)
            round_bar.update(total=None, progress=0)

        if event.message:
            self.query_one("#gen_log", RichLog).write(f"[dim]{label}[/dim]  {event.message}")

    def _update_live_text(self, event: ProgressEvent) -> None:
        panel = self.query_one("#live_panel", VerticalScroll)
        live_text = self.query_one("#live_text", Static)
        panel.border_title = self._live_title(event)

        if event.kind == "text_reset":
            # Keep showing the previous segment's content until new content arrives.
            self._live_buffer = ""
            return

        if event.kind == "text_fragment":
            self._live_buffer += event.text
            if event.text_scope == "script":
                content = _best_effort_dialogue_markup(self._live_buffer)
                if content:  # only replace when something is parseable
                    live_text.update(content)
                    panel.scroll_end(animate=False)
            else:
                live_text.update(escape(self._live_buffer))
                panel.scroll_end(animate=False)
            return

        if event.kind == "text_replace":
            self._live_buffer = event.text
            if event.text_scope == "script":
                live_text.update(_format_turns_markup(event.text))
            else:
                live_text.update(escape(event.text))
            panel.scroll_end(animate=False)

    @staticmethod
    def _live_title(event: ProgressEvent) -> str:
        if event.text_scope == "script":
            segment = f"Segment {event.seg}" if event.seg else "Script"
            if event.segment_title:
                segment = f"{segment}: {event.segment_title}"
            return f"Live text - {segment}"
        if event.text_scope == "outline":
            return "Live text - Outline"
        return "Live text"

    def _show_synthesis_turn(self, event: ProgressEvent) -> None:
        panel = self.query_one("#live_panel", VerticalScroll)
        live_text = self.query_one("#live_text", Static)
        seg_label = f"Segment {event.seg}" if event.seg else "Synthesizing"
        panel.border_title = f"Synthesizing — {seg_label}"
        color = _SPEAKER_COLORS.get(event.speaker or "", "white")
        lines: list[str] = [f"[bold {color}]{escape(event.speaker or '')}[/bold {color}]"]
        if event.delivery_instruction:
            lines.append(f"[dim italic]{escape(event.delivery_instruction)}[/dim italic]")
        if event.text:
            lines.append(f"\n{escape(event.text)}")
        live_text.update("\n".join(lines))

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


_SPEAKER_COLORS = {"Host A": "cyan", "Host B": "magenta"}


def _format_turns_markup(plain: str) -> str:
    """Convert 'Speaker: text\\n\\nSpeaker: text' plain format to Rich markup."""
    parts: list[str] = []
    for block in plain.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if ": " in block:
            speaker, dialogue = block.split(": ", 1)
            speaker = speaker.strip()
            color = _SPEAKER_COLORS.get(speaker, "white")
            parts.append(f"[bold {color}]{escape(speaker)}[/bold {color}]  {escape(dialogue)}")
        else:
            parts.append(escape(block))
    return "\n\n".join(parts)


def _best_effort_dialogue_markup(buffer: str) -> str:
    lines: list[str] = []
    turn_pattern = re.compile(
        r'\{\s*"speaker"\s*:\s*"(?P<speaker>Host A|Host B)"(?P<body>.*?)(?=\{\s*"speaker"|\]\s*\}|\Z)',
        re.DOTALL,
    )
    for turn_match in turn_pattern.finditer(buffer):
        text_match = re.search(r'"text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)', turn_match.group("body"))
        if text_match is None:
            continue
        text = _decode_jsonish_string(text_match.group("text"))
        if not text:
            continue
        speaker = turn_match.group("speaker")
        color = _SPEAKER_COLORS.get(speaker, "white")
        lines.append(f"[bold {color}]{speaker}[/bold {color}]  {escape(text)}")
    return "\n\n".join(lines)


def _decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\"", '"').replace(r"\\", "\\")
