import numpy as np

from .models import DialogueTurn, GenerationRequest


HIGH_QUALITY_TOTAL_STEPS = 100
NATURAL_SPEECH_SPEED = 1.0


class SupertonicSynthesizer:
    sample_rate = 44_100

    def __init__(self) -> None:
        from supertonic import TTS

        self.tts = TTS(auto_download=True)

    def synthesize_turn(self, turn: DialogueTurn, request: GenerationRequest) -> np.ndarray:
        voice_name = request.host_a_voice if turn.speaker == "Host A" else request.host_b_voice
        style = self.tts.get_voice_style(voice_name=voice_name)
        wav, _duration = self.tts.synthesize(
            text=turn.text,
            lang=request.language,
            voice_style=style,
            total_steps=HIGH_QUALITY_TOTAL_STEPS,
            speed=NATURAL_SPEECH_SPEED,
        )
        return np.asarray(wav, dtype=np.float32).reshape(-1)


def list_installed_voices() -> list[str] | None:
    """Return locally installed Supertonic voices, or None if unavailable.

    Reads only the existing model cache; it never triggers a download. Returns
    None when the model has not been downloaded yet or supertonic is missing.
    """
    try:
        from supertonic.loader import get_cache_dir, list_available_voice_style_names

        names = list_available_voice_style_names(get_cache_dir())
    except Exception:
        return None
    return names or None
