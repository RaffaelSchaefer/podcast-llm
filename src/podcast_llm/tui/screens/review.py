from pydantic import ValidationError
from textual.app import ComposeResult
from textual.widgets import Markdown

from ...presets import preset_by_id
from .base import WizardScreen

LANGUAGE_NAMES = {"en": "English", "de": "German", "na": "Language agnostic"}


class ReviewScreen(WizardScreen):
    STEP_INDEX = 3
    CARD_TITLE = "Review"
    NEXT_LABEL = "Generate"

    def compose_body(self) -> ComposeResult:
        yield Markdown("", id="review_summary")

    def populate(self) -> None:
        self.query_one("#review_summary", Markdown).update(self._summary())

    def action_next(self) -> None:
        try:
            self.commit(strict=True)
            self.app.draft.to_request()
        except (ValueError, ValidationError) as exc:
            self.set_status(str(exc))
            return
        self.app.start_generation()

    def _summary(self) -> str:
        draft = self.app.draft
        preset = preset_by_id(draft.preset_id)
        style = preset.label if preset else "Custom"
        language = LANGUAGE_NAMES.get(draft.language, draft.language)
        request = draft.to_request()
        endpoint = request.lmstudio_host or "default LM Studio host (localhost:1234)"
        model = request.lmstudio_model or "currently loaded model"

        lines = [f"### {len(draft.source_paths)} source file(s)"]
        lines += [f"- `{path}`" for path in draft.source_paths] or ["- _none selected_"]
        lines += [
            "",
            "| Setting | Value |",
            "| --- | --- |",
            f"| Style | {style} |",
            f"| Language | {language} |",
            f"| Duration | {draft.duration_minutes} min |",
            f"| Host A voice | {draft.host_a_voice} |",
            f"| Host B voice | {draft.host_b_voice} |",
            f"| TTS mode | {draft.tts_mode} |",
            f"| TTS model | {draft.qwen_tts_model} |",
            f"| Export | {draft.export_format.upper()} |",
            f"| Quiet dynamic background music | {'Yes' if draft.enable_background_music else 'No'} |",
            f"| LLM endpoint | {endpoint} |",
            f"| Model | {model} |",
        ]
        if draft.custom_instructions:
            lines += ["", "**Instructions**", "", f"> {draft.custom_instructions}"]
        return "\n".join(lines)
