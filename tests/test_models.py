from pathlib import Path

import pytest

from podcast_llm.models import GenerationRequest


def test_generation_request_applies_defaults(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")

    request = GenerationRequest(source_paths=[source], language="en", duration_minutes=8)

    assert request.host_a_voice == "M1"
    assert request.host_b_voice == "F1"
    assert request.llm_provider == "lmstudio"
    assert request.lmstudio_model is None
    assert request.lmstudio_host is None
    assert request.export_format == "wav"
    assert request.custom_instructions == ""


def test_generation_request_reads_lmstudio_host_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    monkeypatch.setenv("PODCAST_LLM_HOST", "localhost:1234")

    request = GenerationRequest(source_paths=[source], language="en", duration_minutes=8)

    assert request.lmstudio_host == "localhost:1234"


def test_generation_request_rejects_duration_outside_one_hour(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")

    with pytest.raises(ValueError, match="less than or equal to 60"):
        GenerationRequest(source_paths=[source], language="en", duration_minutes=61)
