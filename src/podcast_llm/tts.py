from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from .models import DialogueTurn, GenerationRequest, TtsMode


QWEN_CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_DEFAULT_SAMPLE_RATE = 24_000

QWEN_LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "na": "Auto",
}


def preflight_qwen_runtime() -> None:
    """Import Qwen TTS before the generation worker starts.

    qwen-tts imports audio dependencies that may probe the external SoX binary
    during import. In some terminal UI worker environments, that subprocess
    startup can surface as the opaque ``bad value(s) in fds_to_keep`` error.
    Running the import on the main thread either avoids the worker-fd issue or
    lets us show an actionable startup error.
    """
    try:
        import_module("qwen_tts")
    except ValueError as exc:
        if "bad value(s) in fds_to_keep" in str(exc):
            raise RuntimeError(
                "Qwen TTS startup failed while probing audio dependencies. "
                "This is usually triggered by the SoX check inside qwen-tts/librosa. "
                "Install SoX and restart the app from a normal terminal, then try generation again. "
                f"Original error: {exc}"
            ) from exc
        raise


class QwenSynthesizer:
    sample_rate = QWEN_DEFAULT_SAMPLE_RATE

    def __init__(self, request: GenerationRequest) -> None:
        from qwen_tts import Qwen3TTSModel
        import torch

        kwargs = _qwen_load_kwargs(request.tts_mode, torch)
        self.model = Qwen3TTSModel.from_pretrained(request.qwen_tts_model, **kwargs)

    def synthesize_turn(self, turn: DialogueTurn, request: GenerationRequest) -> np.ndarray:
        speaker = request.host_a_voice if turn.speaker == "Host A" else request.host_b_voice
        wavs, sample_rate = self.model.generate_custom_voice(
            text=turn.text,
            language=qwen_language_name(request.language),
            speaker=speaker,
            instruct=turn.delivery_instruction,
            non_streaming_mode=True,
        )
        self.sample_rate = int(sample_rate)
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1)


def qwen_language_name(language: str) -> str:
    return QWEN_LANGUAGE_NAMES.get(language.strip().lower(), "Auto")


def _qwen_load_kwargs(mode: TtsMode, torch: Any) -> dict[str, Any]:
    if mode == "cuda":
        return _cuda_load_kwargs(torch)
    if mode == "apple":
        return _apple_load_kwargs(torch)
    if torch.cuda.is_available():
        return _cuda_load_kwargs(torch)
    if torch.backends.mps.is_available():
        return _apple_load_kwargs(torch)
    raise RuntimeError(
        "No supported TTS accelerator found. Use tts_mode='cuda' on a CUDA machine "
        "or tts_mode='apple' on an Apple Silicon machine with PyTorch MPS available."
    )


def _cuda_load_kwargs(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA TTS mode requested, but torch.cuda.is_available() is false.")
    kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
    if find_spec("flash_attn") is not None:
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def _apple_load_kwargs(torch: Any) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple TTS mode requested, but torch.backends.mps.is_available() is false.")
    return {"device_map": "mps", "dtype": torch.float16}
