import re
from pathlib import Path

from markitdown import MarkItDown

from .models import GenerationRequest, ParsedSource, ParsedSources


class SourceParseError(RuntimeError):
    pass


def _normalize_markdown(text: str) -> str:
    text = re.sub(r"\r+\n", "\n", text)
    return text.replace("\r", "\n").strip()


def _ocr_markitdown_for_request(request: GenerationRequest | None) -> MarkItDown | None:
    if request is None:
        return None

    # LM Studio is driven through the ``lmstudio`` SDK, which MarkItDown can't
    # use as an ``llm_client``. We still enable plugins so the markitdown-ocr
    # plugin can extract text from scanned documents without an LLM.
    try:
        return MarkItDown(enable_plugins=True)
    except Exception:
        return None


def _markdown_from_result(result: object) -> str | None:
    markdown = getattr(result, "markdown", None)
    if markdown:
        return str(markdown)
    text_content = getattr(result, "text_content", None)
    if text_content:
        return str(text_content)
    return None


def _convert_with_markitdown(path: Path, request: GenerationRequest | None = None) -> str:
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8")

    failures: list[str] = []
    ocr_converter = _ocr_markitdown_for_request(request)
    converters = [ocr_converter] if ocr_converter is not None else []
    converters.append(MarkItDown)

    for converter_ref in converters:
        converter = converter_ref() if converter_ref is MarkItDown else converter_ref
        try:
            markdown = _markdown_from_result(converter.convert(str(path)))
        except Exception as exc:
            failures.append(str(exc))
            continue
        if markdown:
            return markdown
        failures.append("returned no text")

    if failures:
        raise SourceParseError(f"MarkItDown could not convert {path}: {'; '.join(failures)}")
    raise SourceParseError(f"MarkItDown returned no text for {path}")


def parse_sources(paths: list[Path], request: GenerationRequest | None = None) -> ParsedSources:
    sources: list[ParsedSource] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise SourceParseError(f"Source file does not exist: {path}")
        if not path.is_file():
            raise SourceParseError(f"Source path is not a file: {path}")

        try:
            markdown = _normalize_markdown(_convert_with_markitdown(path, request=request))
        except SourceParseError:
            raise
        except Exception as exc:
            raise SourceParseError(f"Could not parse {path}: {exc}") from exc

        sources.append(ParsedSource(path=path, markdown=markdown))

    combined = "\n\n".join(f"# Source: {source.path.name}\n\n{source.markdown}" for source in sources)
    return ParsedSources(markdown=combined, sources=sources)


class MarkItDownParser:
    def parse(self, paths: list[Path], request: GenerationRequest | None = None) -> ParsedSources:
        return parse_sources(paths, request=request)
