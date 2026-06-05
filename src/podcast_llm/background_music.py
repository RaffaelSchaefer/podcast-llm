from dataclasses import dataclass
import hashlib
import logging
import re
from typing import Any

import numpy as np

from .models import DialogueTurn, OutlineSegment


ACE_STEP_MODEL_ID = "ACE-Step/acestep-v15-base"
BASE_BACKGROUND_MUSIC_PROMPT = (
    "quiet instrumental background music for a podcast, warm ambient piano, soft pads, "
    "slow tempo, no vocals, unobtrusive, calm, gentle, minimal drums, loopable"
)
DEFAULT_MUSIC_GAIN = 0.08
DEFAULT_FADE_SECONDS = 0.75

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SectionMusicSpec:
    index: int
    title: str
    prompt: str
    duration_seconds: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "prompt": self.prompt,
            "duration_seconds": self.duration_seconds,
        }


class AceStepMusicGenerator:
    """Lazy ACE-Step wrapper for section background music generation."""

    model_id = ACE_STEP_MODEL_ID

    def __init__(self, model_id: str = ACE_STEP_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipe: Any | None = None

    def generate(self, prompt: str, duration_seconds: float, sample_rate: int) -> np.ndarray:
        pipe = self._pipeline()
        try:
            output = pipe(
                prompt=prompt,
                lyrics="",
                audio_duration=max(10.0, float(duration_seconds)),
                output_type="np",
            )
        except Exception as exc:
            if isinstance(pipe, _ProceduralMusicPipeline):
                raise
            logger.warning("ACE-Step generation failed; using procedural background music fallback: %s", exc)
            pipe = _ProceduralMusicPipeline()
            self._pipe = pipe
            output = pipe(
                prompt=prompt,
                lyrics="",
                audio_duration=max(10.0, float(duration_seconds)),
                output_type="np",
            )
        audio = _extract_pipeline_audio(output)
        generated_rate = _extract_pipeline_sample_rate(output) or getattr(pipe, "sample_rate", None) or sample_rate
        audio = to_mono_float32(audio)
        if generated_rate != sample_rate:
            audio = resample_audio(audio, generated_rate, sample_rate)
        return fit_audio_to_length(audio, round(duration_seconds * sample_rate))

    def _pipeline(self) -> Any:
        if self._pipe is None:
            try:
                self._pipe = _load_ace_step_pipeline(self.model_id)
            except Exception as exc:
                logger.warning("ACE-Step pipeline failed to load; using procedural background music fallback: %s", exc)
                self._pipe = _ProceduralMusicPipeline()
        return self._pipe


def build_section_prompt(segment: OutlineSegment, turns: list[DialogueTurn]) -> str:
    text = " ".join([segment.title, segment.summary, *(turn.text for turn in turns)])
    moods = _section_moods(text)
    context = _section_context(segment.title, segment.summary)
    return (
        f"{BASE_BACKGROUND_MUSIC_PROMPT}. "
        f"Mood: {', '.join(moods)}. "
        f"Match a section about {context}. "
        "Keep the music subtle and non-distracting."
    )


def fit_audio_to_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        return np.zeros((0,), dtype=np.float32)
    source = to_mono_float32(audio)
    if len(source) == 0:
        return np.zeros((target_samples,), dtype=np.float32)
    repeats = int(np.ceil(target_samples / len(source)))
    return np.tile(source, repeats)[:target_samples].astype(np.float32)


def apply_edge_fades(
    audio: np.ndarray,
    sample_rate: int,
    fade_seconds: float = DEFAULT_FADE_SECONDS,
) -> np.ndarray:
    faded = to_mono_float32(audio).copy()
    fade_samples = min(round(sample_rate * fade_seconds), len(faded) // 2)
    if fade_samples <= 1:
        return faded
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    faded[:fade_samples] *= fade_in
    faded[-fade_samples:] *= fade_out
    return faded.astype(np.float32)


def mix_background_music(
    speech: np.ndarray,
    music: np.ndarray,
    gain: float = DEFAULT_MUSIC_GAIN,
) -> np.ndarray:
    speech_audio = to_mono_float32(speech)
    music_audio = normalize_peak(fit_audio_to_length(music, len(speech_audio)))
    mixed = speech_audio + (music_audio * gain)
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    normalized = to_mono_float32(audio)
    peak = peak_amplitude(normalized)
    if peak <= 0.0:
        return normalized
    return (normalized / peak).astype(np.float32)


def peak_amplitude(audio: np.ndarray) -> float:
    source = to_mono_float32(audio)
    return float(np.max(np.abs(source))) if len(source) else 0.0


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array.astype(np.float32)
    if array.shape[0] in (1, 2) and array.shape[1] > array.shape[0]:
        return np.mean(array, axis=0).astype(np.float32)
    return np.mean(array, axis=1).astype(np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    source = to_mono_float32(audio)
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate or len(source) == 0:
        return source
    target_len = max(1, round(len(source) * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, len(source), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, target_len, endpoint=True)
    return np.interp(target_positions, source_positions, source).astype(np.float32)


def _section_moods(text: str) -> list[str]:
    lowered = text.lower()
    mood_rules = [
        ("light tension", ("risk", "concern", "problem", "challenge", "tradeoff", "serious")),
        ("analytical", ("compare", "decision", "technical", "evidence", "data", "practical")),
        ("optimistic", ("opportunity", "benefit", "success", "useful", "future", "growth")),
        ("reflective", ("history", "learn", "meaning", "context", "lesson", "why")),
    ]
    moods = [mood for mood, keywords in mood_rules if any(keyword in lowered for keyword in keywords)]
    if "curious" not in moods:
        moods.insert(0, "curious")
    return moods[:3]


def _section_context(title: str, summary: str) -> str:
    raw = " ".join([title, summary]).strip()
    words = re.findall(r"[A-Za-z0-9]+", raw)
    context = " ".join(words[:12])
    return context or "the current podcast section"


def _extract_pipeline_audio(output: Any) -> np.ndarray:
    if isinstance(output, dict) and "audio" in output:
        return np.asarray(output["audio"], dtype=np.float32)
    if isinstance(output, dict) and "audios" in output:
        return np.asarray(output["audios"], dtype=np.float32)
    if hasattr(output, "audios"):
        return np.asarray(output.audios, dtype=np.float32)
    if isinstance(output, list) and output:
        return _extract_pipeline_audio(output[0])
    return np.asarray(output, dtype=np.float32)


def _extract_pipeline_sample_rate(output: Any) -> int | None:
    if isinstance(output, dict):
        for key in ("sampling_rate", "sample_rate"):
            if key in output and output[key]:
                return int(output[key])
    for key in ("sampling_rate", "sample_rate"):
        value = getattr(output, key, None)
        if value:
            return int(value)
    if isinstance(output, list) and output:
        return _extract_pipeline_sample_rate(output[0])
    return None


def _load_ace_step_pipeline(model_id: str):
    try:
        import torch
        from diffusers import AceStepPipeline
    except ImportError as exc:
        raise ImportError("diffusers with AceStepPipeline support is required for ACE-Step music generation.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = AceStepPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    return pipe.to(device)


class _ProceduralMusicPipeline:
    sample_rate = 48_000

    def __call__(self, prompt: str, audio_duration: float, **_: Any) -> dict[str, np.ndarray]:
        samples = max(1, round(float(audio_duration) * self.sample_rate))
        t = np.arange(samples, dtype=np.float32) / self.sample_rate
        seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        root = 55.0 * (2 ** ((seed % 12) / 12))
        chord = np.zeros_like(t)
        for index, ratio in enumerate((1.0, 1.5, 2.0, 2.5)):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            lfo = 0.55 + 0.45 * np.sin(2.0 * np.pi * (0.025 + index * 0.01) * t + phase)
            chord += (0.08 / (index + 1)) * np.sin(2.0 * np.pi * root * ratio * t + phase) * lfo
        noise = rng.normal(0.0, 0.006, samples).astype(np.float32)
        audio = apply_edge_fades((chord + noise).astype(np.float32), self.sample_rate, fade_seconds=1.5)
        return {"audios": audio.reshape(1, 1, -1), "sample_rate": self.sample_rate}


def _transformers_pipeline(*args, **kwargs):
    from transformers import pipeline

    return pipeline(*args, **kwargs)
