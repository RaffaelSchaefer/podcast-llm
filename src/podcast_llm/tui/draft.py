from dataclasses import dataclass, field
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
    host_a_voice: str = "M1"
    host_b_voice: str = "F1"
    custom_instructions: str = ""
    lmstudio_host: str | None = None
    export_format: str = "wav"
    preset_id: str | None = None

    def to_request(self) -> GenerationRequest:
        # Only pass an explicit host override; leaving it out lets the
        # GenerationRequest default read PODCAST_LLM_HOST from the environment.
        overrides: dict = {}
        if self.lmstudio_host:
            overrides["lmstudio_host"] = self.lmstudio_host
        return GenerationRequest(
            source_paths=list(self.source_paths),
            language=self.language,
            duration_minutes=self.duration_minutes,
            host_a_voice=self.host_a_voice or "M1",
            host_b_voice=self.host_b_voice or "F1",
            custom_instructions=self.custom_instructions,
            export_format=self.export_format,
            **overrides,
        )
