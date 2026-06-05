from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Checkbox, Input, Label, Select

from ...llm import list_available_models
from ...voices import QWEN_VOICE_NAMES, voice_select_options
from .base import WizardScreen

EXPORT_OPTIONS = [("WAV", "wav"), ("MP3 (needs ffmpeg)", "mp3")]
TTS_MODE_OPTIONS = [
    ("Auto-detect", "auto"),
    ("CUDA (NVIDIA)", "cuda"),
    ("Apple Silicon (MPS)", "apple"),
]
CURRENT_MODEL = "__current_lmstudio_model__"
CURRENT_MODEL_LABEL = "Currently loaded model / env default"


class VoicesScreen(WizardScreen):
    STEP_INDEX = 2
    CARD_TITLE = "Voices & Engine"

    def compose_body(self) -> ComposeResult:
        yield Label("Host A voice", classes="section-label")
        yield Select(voice_select_options(), value="Aiden", allow_blank=False, id="host_a")
        yield Label("Host B voice", classes="section-label")
        yield Select(voice_select_options(), value="Serena", allow_blank=False, id="host_b")
        yield Label("TTS hardware mode", classes="section-label")
        yield Select(TTS_MODE_OPTIONS, value="auto", allow_blank=False, id="tts_mode")
        yield Label("Export format", classes="section-label")
        yield Select(EXPORT_OPTIONS, value="wav", allow_blank=False, id="export")
        yield Checkbox("Quiet dynamic background music", id="background_music")
        yield Label("LM Studio host (optional, blank uses the env / default)", classes="section-label")
        yield Input(placeholder="host:port, e.g. localhost:1234", id="lmstudio_host")
        yield Label("LM Studio model", classes="section-label")
        yield Select(
            [(CURRENT_MODEL_LABEL, CURRENT_MODEL)],
            value=CURRENT_MODEL,
            allow_blank=False,
            id="lmstudio_model",
        )
        with Horizontal(id="llm_actions"):
            yield Button("Refresh models", id="test_connection")

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_models()

    def populate(self) -> None:
        draft = self.app.draft
        self._set_voice("host_a", draft.host_a_voice)
        self._set_voice("host_b", draft.host_b_voice)
        self.query_one("#tts_mode", Select).value = draft.tts_mode
        self.query_one("#export", Select).value = draft.export_format
        self.query_one("#background_music", Checkbox).value = draft.enable_background_music
        self.query_one("#lmstudio_host", Input).value = draft.lmstudio_host or ""
        self._set_model_options([], draft.lmstudio_model)

    def commit(self, strict: bool) -> None:
        draft = self.app.draft
        draft.host_a_voice = self._read_voice("host_a")
        draft.host_b_voice = self._read_voice("host_b")
        draft.tts_mode = str(self.query_one("#tts_mode", Select).value or "auto")
        draft.export_format = str(self.query_one("#export", Select).value or "wav")
        draft.enable_background_music = bool(self.query_one("#background_music", Checkbox).value)
        draft.lmstudio_host = self.query_one("#lmstudio_host", Input).value.strip() or None
        model = self.query_one("#lmstudio_model", Select).value
        draft.lmstudio_model = None if model in (None, CURRENT_MODEL) else str(model)

    # --- helpers ---
    def _set_voice(self, select_id: str, value: str) -> None:
        select = self.query_one(f"#{select_id}", Select)
        if value in QWEN_VOICE_NAMES:
            select.value = value
        else:
            select.value = QWEN_VOICE_NAMES[0]

    def _read_voice(self, select_id: str) -> str:
        return str(self.query_one(f"#{select_id}", Select).value or QWEN_VOICE_NAMES[0])

    # --- LLM connection test ---
    @on(Button.Pressed, "#test_connection")
    def _on_test_connection(self, event: Button.Pressed) -> None:
        self._refresh_models()

    def _refresh_models(self) -> None:
        self.commit(strict=False)
        request = self.app.draft.to_request()
        host = request.lmstudio_host
        target = host or "the default LM Studio host"
        self.set_status(f"Loading LM Studio models from {target}...")
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
        current = self.query_one("#lmstudio_model", Select).value
        selected = None if current in (None, CURRENT_MODEL) else str(current)
        self._set_model_options(models, selected)
        if not models:
            self.set_status("Connected, but the endpoint reported no models.")
            return
        preview = ", ".join(models[:5])
        if len(models) > 5:
            preview += f", ... (+{len(models) - 5} more)"
        self.set_status(f"Connected - {len(models)} model{'s' if len(models) != 1 else ''}: {preview}")

    def _set_model_options(self, models: list[str], selected: str | None) -> None:
        model_select = self.query_one("#lmstudio_model", Select)
        keys = list(dict.fromkeys([model for model in models if model]))
        if selected and selected not in keys:
            keys.insert(0, selected)
        options = [(CURRENT_MODEL_LABEL, CURRENT_MODEL)] + [(key, key) for key in sorted(keys)]
        model_select.set_options(options)
        model_select.value = selected or CURRENT_MODEL
