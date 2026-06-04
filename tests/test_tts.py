import sys
from types import SimpleNamespace

import numpy as np

from podcast_llm.models import DialogueTurn, GenerationRequest
from podcast_llm.tts import SupertonicSynthesizer


class FakeTTS:
    instances = []

    def __init__(self, auto_download: bool) -> None:
        self.auto_download = auto_download
        self.calls = []
        FakeTTS.instances.append(self)

    def get_voice_style(self, voice_name: str) -> str:
        return f"style:{voice_name}"

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return np.array([[0.1, 0.2]], dtype=np.float32), np.array([0.2], dtype=np.float32)


def test_supertonic_synthesizer_uses_max_quality_settings(monkeypatch, tmp_path) -> None:
    FakeTTS.instances = []
    monkeypatch.setitem(sys.modules, "supertonic", SimpleNamespace(TTS=FakeTTS))
    request = GenerationRequest(
        source_paths=[tmp_path / "notes.md"],
        language="en",
        duration_minutes=1,
        host_a_voice="M1",
        host_b_voice="F1",
    )
    turn = DialogueTurn(speaker="Host B", text="Use the best available synthesis quality.")

    audio = SupertonicSynthesizer().synthesize_turn(turn, request)

    call = FakeTTS.instances[0].calls[0]
    assert call["voice_style"] == "style:F1"
    assert call["total_steps"] == 100
    assert call["speed"] == 1.0
    np.testing.assert_array_equal(audio, np.array([0.1, 0.2], dtype=np.float32))
