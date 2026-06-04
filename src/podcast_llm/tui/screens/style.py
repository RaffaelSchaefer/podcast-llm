from textual.app import ComposeResult
from textual.widgets import Input, Label, RadioButton, RadioSet, Select, TextArea

from ...presets import CUSTOM_PRESET_ID, PRESETS, preset_by_id
from .base import WizardScreen

LANGUAGE_OPTIONS = [("English", "en"), ("German", "de"), ("Language agnostic", "na")]


class StyleScreen(WizardScreen):
    STEP_INDEX = 1
    CARD_TITLE = "Style & Format"

    def compose_body(self) -> ComposeResult:
        yield Label("Pick a style preset to set the tone (you can still tweak everything):", classes="section-label")
        with RadioSet(id="preset"):
            yield RadioButton("Custom", value=True, id=f"preset_{CUSTOM_PRESET_ID}")
            for preset in PRESETS:
                yield RadioButton(f"{preset.label} — {preset.description}", id=f"preset_{preset.id}")
        yield Label("Language", classes="section-label")
        yield Select(LANGUAGE_OPTIONS, value="en", allow_blank=False, id="language")
        yield Label("Target duration (minutes, 1–60)", classes="section-label")
        yield Input(value="8", id="duration", type="integer")
        yield Label("Custom instructions (tone, focus, format)", classes="section-label")
        yield TextArea(id="custom_instructions")

    def populate(self) -> None:
        draft = self.app.draft
        target = f"preset_{draft.preset_id}" if preset_by_id(draft.preset_id) else f"preset_{CUSTOM_PRESET_ID}"
        with self.prevent(RadioSet.Changed):
            self.query_one(f"#{target}", RadioButton).value = True
        self.query_one("#language", Select).value = draft.language
        self.query_one("#duration", Input).value = str(draft.duration_minutes)
        self.query_one("#custom_instructions", TextArea).text = draft.custom_instructions

    def commit(self, strict: bool) -> None:
        draft = self.app.draft
        draft.language = str(self.query_one("#language", Select).value or "en")

        duration_raw = self.query_one("#duration", Input).value.strip()
        try:
            duration = int(duration_raw)
        except ValueError:
            if strict:
                raise ValueError("Duration must be a whole number of minutes.") from None
        else:
            if strict and not (1 <= duration <= 60):
                raise ValueError("Duration must be between 1 and 60 minutes.")
            draft.duration_minutes = duration

        draft.custom_instructions = self.query_one("#custom_instructions", TextArea).text.strip()
        pressed = self.query_one("#preset", RadioSet).pressed_button
        preset_id = (pressed.id or "").removeprefix("preset_") if pressed else CUSTOM_PRESET_ID
        draft.preset_id = None if preset_id == CUSTOM_PRESET_ID else preset_id

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        preset_id = (event.pressed.id or "").removeprefix("preset_")
        preset = preset_by_id(preset_id)
        if preset is None:
            return
        self.query_one("#duration", Input).value = str(preset.duration_minutes)
        self.query_one("#custom_instructions", TextArea).text = preset.custom_instructions
        self.set_status(f"Applied the “{preset.label}” preset — edit anything you like.")
