from pathlib import Path

import pytest

from podcast_llm import parser
from podcast_llm.models import GenerationRequest
from podcast_llm.parser import SourceParseError, parse_sources


def test_parse_sources_normalizes_markdown_and_text_files(tmp_path: Path) -> None:
    markdown = tmp_path / "paper.md"
    text = tmp_path / "notes.txt"
    markdown.write_bytes(b"# Paper\r\nMain idea\r\n")
    text.write_bytes(b"Plain notes\r\nSecond line\r\n")

    parsed = parse_sources([markdown, text])

    assert parsed.markdown == (
        "# Source: paper.md\n\n# Paper\nMain idea\n\n"
        "# Source: notes.txt\n\nPlain notes\nSecond line"
    )
    assert [source.path for source in parsed.sources] == [markdown, text]


def test_parse_sources_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SourceParseError, match="Source file does not exist"):
        parse_sources([tmp_path / "missing.pdf"])


def test_parse_sources_uses_plain_markitdown_without_request_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF test")
    markitdown_calls: list[dict] = []

    class FakeResult:
        text_content = "Converted PDF"

    class FakeMarkItDown:
        def __init__(self, **kwargs) -> None:
            markitdown_calls.append(kwargs)

        def convert(self, path: str):
            assert path == str(source)
            return FakeResult()

    monkeypatch.setattr(parser, "MarkItDown", FakeMarkItDown)

    parsed = parse_sources([source])

    assert markitdown_calls == [{}]
    assert parsed.sources[0].markdown == "Converted PDF"


def test_parse_sources_enables_ocr_plugin_with_request_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "scanned.pdf"
    source.write_bytes(b"%PDF test")
    markitdown_calls: list[dict] = []

    class FakeResult:
        text_content = "OCR Markdown"

    class FakeMarkItDown:
        def __init__(self, **kwargs) -> None:
            markitdown_calls.append(kwargs)

        def convert(self, path: str):
            assert path == str(source)
            return FakeResult()

    monkeypatch.setattr(parser, "MarkItDown", FakeMarkItDown)
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=5,
        lmstudio_model="vision-model",
        lmstudio_host="localhost:1234",
    )

    parsed = parse_sources([source], request=request)

    # With a request we enable the markitdown-ocr plugin; no LLM client is wired
    # in (LM Studio is reached through its own SDK, not as a MarkItDown client).
    assert markitdown_calls == [{"enable_plugins": True}]
    assert parsed.sources[0].markdown == "OCR Markdown"


def test_parse_sources_falls_back_when_ocr_converter_cannot_be_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF test")
    markitdown_calls: list[dict] = []

    class FakeResult:
        text_content = "Plain Markdown"

    class FakeMarkItDown:
        def __init__(self, **kwargs) -> None:
            if kwargs.get("enable_plugins"):
                raise RuntimeError("plugin load failed")
            markitdown_calls.append(kwargs)

        def convert(self, path: str):
            assert path == str(source)
            return FakeResult()

    monkeypatch.setattr(parser, "MarkItDown", FakeMarkItDown)
    request = GenerationRequest(source_paths=[source], language="en", duration_minutes=5)

    parsed = parse_sources([source], request=request)

    assert markitdown_calls == [{}]
    assert parsed.sources[0].markdown == "Plain Markdown"
