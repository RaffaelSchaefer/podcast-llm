from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, Label, Select

from ...llm import list_available_models
from ...tts import list_installed_voices
from ...voices import BUILTIN_VOICE_NAMES, voice_select_options
from .base import WizardScreen

EXPORT_OPTIONS = [("WAV", "wav"), ("MP3 (needs ffmpeg)", "mp3")]


class VoicesScreen(WizardScreen):
    STEP_INDEX = 2
    CARD_TITLE = "Voices & Engine"

    def compose_body(self) -> ComposeResult:
        yield Label("Host A voice", classes="section-label")
        yield Select(voice_select_options(), value="M1", allow_blank=False, id="host_a")
        yield Input(placeholder="Custom voice name (optional, overrides the picker)", id="host_a_custom")
        yield Label("Host B voice", classes="section-label")
        yield Select(voice_select_options(), value="F1", allow_blank=False, id="host_b")
        yield Input(placeholder="Custom voice name (optional, overrides the picker)", id="host_b_custom")
        yield Button("Refresh installed voices", id="refresh_voices")
        yield Label("Export format", classes="section-label")
        yield Select(EXPORT_OPTIONS, value="wav", allow_blank=False, id="export")
        yield Label("LM Studio host (optional, blank uses the env / default)", classes="section-label")
        yield Input(placeholder="host:port, e.g. localhost:1234", id="lmstudio_host")
        with Horizontal(id="llm_actions"):
            yield Button("Test connection and list models", id="test_connection")

    def populate(self) -> None:
        draft = self.app.draft
        self._set_voice("host_a", "host_a_custom", draft.host_a_voice)
        self._set_voice("host_b", "host_b_custom", draft.host_b_voice)
        self.query_one("#export", Select).value = draft.export_format
        self.query_one("#lmstudio_host", Input).value = draft.lmstudio_host or ""

    def commit(self, strict: bool) -> None:
        draft = self.app.draft
        draft.host_a_voice = self._read_voice("host_a", "host_a_custom")
        draft.host_b_voice = self._read_voice("host_b", "host_b_custom")
        draft.export_format = str(self.query_one("#export", Select).value or "wav")
        draft.lmstudio_host = self.query_one("#lmstudio_host", Input).value.strip() or None

    # --- helpers ---
    def _set_voice(self, select_id: str, custom_id: str, value: str) -> None:
        select = self.query_one(f"#{select_id}", Select)
        custom = self.query_one(f"#{custom_id}", Input)
        if value in BUILTIN_VOICE_NAMES:
            select.value = value
            custom.value = ""
        else:
            select.value = BUILTIN_VOICE_NAMES[0]
            custom.value = value

    def _read_voice(self, select_id: str, custom_id: str) -> str:
        override = self.query_one(f"#{custom_id}", Input).value.strip()
        if override:
            return override
        return str(self.query_one(f"#{select_id}", Select).value or BUILTIN_VOICE_NAMES[0])

    @on(Button.Pressed, "#refresh_voices")
    def _on_refresh_voices(self, event: Button.Pressed) -> None:
        self.set_status("Looking for installed voices...")
        self._refresh_installed_voices()

    # --- LLM connection test ---
    @on(Button.Pressed, "#test_connection")
    def _on_test_connection(self, event: Button.Pressed) -> None:
        self.commit(strict=False)
        request = self.app.draft.to_request()
        host = request.lmstudio_host
        target = host or "the default LM Studio host"
        self.set_status(f"Connecting to {target}...")
        self._test_connection(host)

    @work(thread=True, exclusive=True)
    def _test_connection(self, host: str | None) -> None:
        try:
            models = list_available_models(host)
        except Exception as exc:  # surface whatever the LM Studio SDK raised
            self.app.call_from_thread(self._show_connection_error, exc)
        else:
            self.app.call_from_thread(self._show_connection_models, models)

    def _show_connection_error(self, exc: Exception) -> None:
        detail = str(exc).strip() or exc.__class__.__name__
        self.set_status(f"Connection failed: {detail}")

    def _show_connection_models(self, models: list[str]) -> None:
        if not models:
            self.set_status("Connected, but the endpoint reported no models.")
            return
        preview = ", ".join(models[:5])
        if len(models) > 5:
            preview += f", ... (+{len(models) - 5} more)"
        self.set_status(f"Connected - {len(models)} model{'s' if len(models) != 1 else ''}: {preview}")

    @work(thread=True, exclusive=True)
    def _refresh_installed_voices(self) -> None:
        names = list_installed_voices()
        self.app.call_from_thread(self._apply_installed_voices, names)

    def _apply_installed_voices(self, names: list[str] | None) -> None:
        if not names:
            self.set_status("No installed voices found yet - using the built-in list.")
            return
        options = voice_select_options(tuple(names))
        for select_id in ("host_a", "host_b"):
            select = self.query_one(f"#{select_id}", Select)
            current = select.value
            select.set_options(options)
            if current in names:
                select.value = current
        self.set_status(f"Loaded {len(names)} installed voice{'s' if len(names) != 1 else ''}.")
