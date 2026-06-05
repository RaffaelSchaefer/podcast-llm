import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ExportFormat = Literal["wav", "mp3"]
TtsMode = Literal["auto", "cuda", "apple"]


class GenerationRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_paths: list[Path]
    language: str
    duration_minutes: int = Field(ge=1, le=60)
    host_a_voice: str = "Aiden"
    host_b_voice: str = "Serena"
    tts_mode: TtsMode = "auto"
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    llm_provider: str = "lmstudio"
    lmstudio_model: str | None = Field(default_factory=lambda: os.getenv("PODCAST_LLM_MODEL") or None)
    lmstudio_host: str | None = Field(default_factory=lambda: os.getenv("PODCAST_LLM_HOST") or None)
    custom_instructions: str = ""
    export_format: ExportFormat = "wav"
    enable_background_music: bool = False
    output_dir: Path = Path("outputs")


class ParsedSource(BaseModel):
    path: Path
    markdown: str


class ParsedSources(BaseModel):
    markdown: str
    sources: list[ParsedSource]


class OutlineSegment(BaseModel):
    title: str
    summary: str
    target_minutes: int = Field(ge=1)
    source_focus: list[str] = Field(default_factory=list)


class EpisodeOutline(BaseModel):
    title: str
    segments: list[OutlineSegment]


class DialogueTurn(BaseModel):
    speaker: Literal["Host A", "Host B"]
    text: str = Field(min_length=1)
    delivery_instruction: str = ""


class GenerationResult(BaseModel):
    output_dir: Path
    transcript_path: Path
    audio_path: Path
    mp3_path: Path | None = None
