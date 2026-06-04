import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import DialogueTurn, EpisodeOutline, GenerationRequest


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "episode"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_transcript(path: Path, outline: EpisodeOutline, turns_by_segment: list[list[DialogueTurn]]) -> None:
    lines = [f"# {outline.title}", ""]
    for segment, turns in zip(outline.segments, turns_by_segment, strict=True):
        lines.extend([f"## {segment.title}", ""])
        for turn in turns:
            lines.append(f"**{turn.speaker}:** {turn.text}")
            lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, sample_rate, subtype="FLOAT")


DEFAULT_SEAM_SILENCE_SECONDS = 0.25


def concat_audio(
    chunks: list[np.ndarray],
    sample_rate: int | None = None,
    seam_silence_seconds: float = DEFAULT_SEAM_SILENCE_SECONDS,
) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.float32)

    flattened = [chunk.reshape(-1).astype(np.float32) for chunk in chunks]
    silence_samples = 0
    if sample_rate is not None and seam_silence_seconds > 0 and len(flattened) > 1:
        silence_samples = round(sample_rate * seam_silence_seconds)
    if silence_samples <= 0:
        return np.concatenate(flattened).astype(np.float32)

    seam_silence = np.zeros((silence_samples,), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, chunk in enumerate(flattened):
        if index:
            joined.append(seam_silence)
        joined.append(chunk)
    return np.concatenate(joined).astype(np.float32)


def maybe_export_mp3(wav_path: Path, mp3_path: Path, request: GenerationRequest) -> Path | None:
    if request.export_format != "mp3":
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "0",
            str(mp3_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp3_path
