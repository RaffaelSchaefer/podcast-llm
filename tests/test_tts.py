import sys
from types import SimpleNamespace

import numpy as np
import pytest

from podcast_llm.models import DialogueTurn, GenerationRequest
from podcast_llm.tts import QWEN_CUSTOM_VOICE_MODEL, QwenSynthesizer, preflight_qwen_runtime


class FakeQwenModel:
    instances = []

    def __init__(self, model_id: str, kwargs: dict) -> None:
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = []
        FakeQwenModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        return cls(model_id, kwargs)

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [np.array([[0.1, 0.2]], dtype=np.float64)], 24_000


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    FakeQwenModel.instances = []
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        cuda=FakeCuda(cuda),
        backends=SimpleNamespace(mps=FakeMps(mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeQwenModel))


def request_for(tmp_path, **overrides) -> GenerationRequest:
    values = {
        "source_paths": [tmp_path / "notes.md"],
        "language": "en",
        "duration_minutes": 1,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def test_qwen_synthesizer_uses_cuda_mode(monkeypatch, tmp_path) -> None:
    install_fake_runtime(monkeypatch, cuda=True, mps=False)
    request = request_for(tmp_path, tts_mode="cuda")

    synthesizer = QwenSynthesizer(request)

    model = FakeQwenModel.instances[0]
    assert model.model_id == QWEN_CUSTOM_VOICE_MODEL
    assert model.kwargs["device_map"] == "cuda:0"
    assert model.kwargs["dtype"] == "bfloat16"
    assert synthesizer.sample_rate == 24_000


def test_qwen_synthesizer_uses_apple_mode(monkeypatch, tmp_path) -> None:
    install_fake_runtime(monkeypatch, cuda=False, mps=True)
    request = request_for(tmp_path, tts_mode="apple")

    QwenSynthesizer(request)

    model = FakeQwenModel.instances[0]
    assert model.kwargs == {"device_map": "mps", "dtype": "float16"}


def test_qwen_synthesizer_auto_prefers_cuda_before_apple(monkeypatch, tmp_path) -> None:
    install_fake_runtime(monkeypatch, cuda=True, mps=True)
    request = request_for(tmp_path)

    QwenSynthesizer(request)

    model = FakeQwenModel.instances[0]
    assert model.kwargs["device_map"] == "cuda:0"


def test_qwen_synthesizer_rejects_unavailable_cuda(monkeypatch, tmp_path) -> None:
    install_fake_runtime(monkeypatch, cuda=False, mps=True)
    request = request_for(tmp_path, tts_mode="cuda")

    with pytest.raises(RuntimeError, match="CUDA TTS mode requested"):
        QwenSynthesizer(request)


def test_qwen_synthesizer_generates_custom_voice(monkeypatch, tmp_path) -> None:
    install_fake_runtime(monkeypatch, cuda=True, mps=False)
    request = request_for(
        tmp_path,
        language="de",
        host_a_voice="Aiden",
        host_b_voice="Serena",
    )
    turn = DialogueTurn(
        speaker="Host B",
        text="Use the best available synthesis quality.",
        delivery_instruction="measured and serious, with a slight pause before the final phrase",
    )

    audio = QwenSynthesizer(request).synthesize_turn(turn, request)

    model = FakeQwenModel.instances[0]
    call = model.calls[0]
    assert call["text"] == "Use the best available synthesis quality."
    assert call["language"] == "German"
    assert call["speaker"] == "Serena"
    assert call["instruct"] == "measured and serious, with a slight pause before the final phrase"
    assert call["non_streaming_mode"] is True
    np.testing.assert_array_equal(audio, np.array([0.1, 0.2], dtype=np.float32))


def test_preflight_qwen_runtime_explains_fds_to_keep(monkeypatch) -> None:
    def fake_import(name: str):
        assert name == "qwen_tts"
        raise ValueError("bad value(s) in fds_to_keep")

    monkeypatch.setattr("podcast_llm.tts.import_module", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        preflight_qwen_runtime()

    assert "Qwen TTS startup failed" in str(exc_info.value)
    assert "SoX" in str(exc_info.value)
    assert "bad value(s) in fds_to_keep" in str(exc_info.value)
