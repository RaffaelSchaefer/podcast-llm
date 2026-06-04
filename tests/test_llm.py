from pathlib import Path

import pytest

from podcast_llm import llm
from podcast_llm.llm import (
    LLMOutputError,
    LMStudioProvider,
    list_available_models,
    provider_for_request,
)
from podcast_llm.models import EpisodeOutline, GenerationRequest, OutlineSegment, ParsedSources


class FakeResult:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """Stand-in for an LM Studio model handle exposing ``respond``."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def respond(self, history, *, response_format=None, **kwargs):
        self.calls.append({"history": history, "response_format": response_format})
        return FakeResult(self.content)


def _last_user_prompt(model: FakeModel) -> str:
    return model.calls[0]["history"]["messages"][-1]["content"]


def make_request(tmp_path: Path) -> GenerationRequest:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    return GenerationRequest(source_paths=[source], language="en", duration_minutes=5)


def test_provider_requests_structured_outline(tmp_path: Path) -> None:
    model = FakeModel(
        '{"title":"Episode","segments":[{"title":"Intro","summary":"Set context",'
        '"target_minutes":2,"source_focus":["notes"]}]}'
    )
    provider = LMStudioProvider(model=model)
    parsed = ParsedSources(markdown="# Notes\n", sources=[])

    outline = provider.make_outline(parsed, make_request(tmp_path))

    assert outline.title == "Episode"
    response_format = model.calls[0]["response_format"]
    assert response_format["type"] == "json"
    schema = response_format["jsonSchema"]
    assert schema["properties"]["segments"]["items"]["required"] == [
        "title",
        "summary",
        "target_minutes",
        "source_focus",
    ]
    assert "Return JSON" in _last_user_prompt(model)


def test_provider_includes_custom_instructions_in_outline_prompt(tmp_path: Path) -> None:
    model = FakeModel(
        '{"title":"Episode","segments":[{"title":"Intro","summary":"Set context",'
        '"target_minutes":2,"source_focus":["notes"]}]}'
    )
    provider = LMStudioProvider(model=model)
    parsed = ParsedSources(markdown="# Notes\n", sources=[])
    request = GenerationRequest(
        source_paths=make_request(tmp_path).source_paths,
        language="en",
        duration_minutes=5,
        custom_instructions="Make this a skeptical executive briefing.",
    )

    provider.make_outline(parsed, request)

    prompt = _last_user_prompt(model)
    assert "Custom podcast instructions:" in prompt
    assert "Make this a skeptical executive briefing." in prompt


def test_provider_omits_empty_custom_instructions_from_outline_prompt(tmp_path: Path) -> None:
    model = FakeModel(
        '{"title":"Episode","segments":[{"title":"Intro","summary":"Set context",'
        '"target_minutes":2,"source_focus":["notes"]}]}'
    )
    provider = LMStudioProvider(model=model)
    parsed = ParsedSources(markdown="# Notes\n", sources=[])

    provider.make_outline(parsed, make_request(tmp_path))

    assert "Custom podcast instructions:" not in _last_user_prompt(model)


def test_provider_rejects_malformed_script_json(tmp_path: Path) -> None:
    model = FakeModel('{"turns":[{"speaker":"Narrator","text":""}]}')
    provider = LMStudioProvider(model=model)
    outline = EpisodeOutline(
        title="Episode",
        segments=[
            OutlineSegment(
                title="Intro",
                summary="Set context",
                target_minutes=2,
                source_focus=["notes"],
            )
        ],
    )

    with pytest.raises(LLMOutputError):
        provider.make_script_segment(outline.segments[0], "", make_request(tmp_path))


def test_provider_raises_on_empty_response(tmp_path: Path) -> None:
    model = FakeModel("")
    provider = LMStudioProvider(model=model)
    parsed = ParsedSources(markdown="# Notes\n", sources=[])

    with pytest.raises(LLMOutputError, match="empty response"):
        provider.make_outline(parsed, make_request(tmp_path))


def test_provider_includes_custom_instructions_in_script_prompt(tmp_path: Path) -> None:
    model = FakeModel('{"turns":[{"speaker":"Host A","text":"Welcome."}]}')
    provider = LMStudioProvider(model=model)
    segment = OutlineSegment(
        title="Intro",
        summary="Set context",
        target_minutes=2,
        source_focus=["notes"],
    )
    request = GenerationRequest(
        source_paths=make_request(tmp_path).source_paths,
        language="en",
        duration_minutes=5,
        custom_instructions="Make the hosts disagree before converging.",
    )

    provider.make_script_segment(segment, "", request)

    prompt = _last_user_prompt(model)
    assert "Custom podcast instructions:" in prompt
    assert "Make the hosts disagree before converging." in prompt


def test_provider_requests_structured_script(tmp_path: Path) -> None:
    model = FakeModel('{"turns":[{"speaker":"Host A","text":"Welcome."}]}')
    provider = LMStudioProvider(model=model)
    segment = OutlineSegment(
        title="Intro",
        summary="Set context",
        target_minutes=2,
        source_focus=["notes"],
    )

    provider.make_script_segment(segment, "", make_request(tmp_path))

    response_format = model.calls[0]["response_format"]
    assert response_format["type"] == "json"
    turns_schema = response_format["jsonSchema"]["properties"]["turns"]
    assert turns_schema["items"]["properties"]["speaker"]["enum"] == ["Host A", "Host B"]


def test_provider_for_request_loads_named_model_on_custom_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: dict = {}

    class FakeLlmNamespace:
        def model(self, model_key=None):
            created["model_key"] = model_key
            return FakeModel('{"title":"Episode","segments":[]}')

    class FakeClient:
        def __init__(self, host=None) -> None:
            created["host"] = host
            self.llm = FakeLlmNamespace()

    monkeypatch.setattr(llm.lms, "Client", FakeClient)
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=5,
        lmstudio_host="localhost:1234",
        lmstudio_model="qwen/qwen3.6-35b-a3b",
    )

    provider = provider_for_request(request)

    assert isinstance(provider, LMStudioProvider)
    assert created == {"host": "localhost:1234", "model_key": "qwen/qwen3.6-35b-a3b"}


def test_provider_for_request_uses_default_host_and_loaded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: dict = {}

    class FakeLlmNamespace:
        def model(self, model_key=None):
            created["model_key"] = model_key
            return FakeModel('{"title":"Episode","segments":[]}')

    class FakeClient:
        def __init__(self, host=None) -> None:
            created["host"] = host
            self.llm = FakeLlmNamespace()

    monkeypatch.setattr(llm.lms, "Client", FakeClient)
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(source_paths=[source], language="en", duration_minutes=5)

    provider_for_request(request)

    # No host -> let the SDK find its default; no model -> reuse the loaded model.
    assert created == {"host": None, "model_key": None}


def test_provider_for_request_rejects_unknown_provider(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=5,
        llm_provider="something-else",
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        provider_for_request(request)


def test_list_available_models_returns_sorted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict = {}

    class FakeModel:
        def __init__(self, model_key: str) -> None:
            self.model_key = model_key

    class FakeLlmNamespace:
        def list_downloaded(self):
            return [FakeModel("zephyr"), FakeModel("llama-3"), FakeModel("phi-3")]

    class FakeClient:
        def __init__(self, host=None) -> None:
            created["host"] = host
            self.llm = FakeLlmNamespace()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm.lms, "Client", FakeClient)

    models = list_available_models(host="localhost:1234")

    assert models == ["llama-3", "phi-3", "zephyr"]
    assert created == {"host": "localhost:1234"}
