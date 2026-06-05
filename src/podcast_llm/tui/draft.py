from dataclasses import dataclass, field
import os
from pathlib import Path

from ..models import GenerationRequest


@dataclass
class PodcastDraft:
    """Mutable working config shared across wizard steps.

    Each step reads it in ``populate()`` and writes it in ``commit()``.
    ``to_request()`` produces the validated pydantic model the pipeline needs.
    """

    source_paths: list[Path] = field(default_factory=list)
    language: str = "en"
    duration_minutes: int = 8
    host_a_voice: str = "Aiden"
    host_b_voice: str = "Serena"
    tts_mode: str = "auto"
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    custom_instructions: str = ""
    lmstudio_host: str | None = None
    lmstudio_model: str | None = field(default_factory=lambda: os.getenv("PODCAST_LLM_MODEL") or None)
    export_format: str = "wav"
    enable_background_music: bool = False
    preset_id: str | None = None

    def to_request(self) -> GenerationRequest:
        # Only pass an explicit host override; leaving it out lets the
        # GenerationRequest default read PODCAST_LLM_HOST from the environment.
        overrides: dict = {}
        if self.lmstudio_host:
            overrides["lmstudio_host"] = self.lmstudio_host
        if self.lmstudio_model:
            overrides["lmstudio_model"] = self.lmstudio_model
        return GenerationRequest(
            source_paths=list(self.source_paths),
            language=self.language,
            duration_minutes=self.duration_minutes,
            host_a_voice=self.host_a_voice or "Aiden",
            host_b_voice=self.host_b_voice or "Serena",
            tts_mode=self.tts_mode,
            qwen_tts_model=self.qwen_tts_model,
            custom_instructions=self.custom_instructions,
            export_format=self.export_format,
            enable_background_music=self.enable_background_music,
            **overrides,
        )
