from pathlib import Path

import pytest

from podcast_llm import llm
from podcast_llm.llm import (
    DEFAULT_LM_STUDIO_HOST,
    LLMOutputError,
    LMStudioProvider,
    list_available_models,
    provider_for_request,
)
from podcast_llm.models import EpisodeOutline, GenerationRequest, OutlineSegment, ParsedSources


class FakeResult:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeFragment:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeStream:
    def __init__(self, fragments: list[str], on_prediction_fragment=None) -> None:
        self.fragments = fragments
        self.on_prediction_fragment = on_prediction_fragment

    def __iter__(self):
        for content in self.fragments:
            fragment = FakeFragment(content)
            if self.on_prediction_fragment is not None:
                self.on_prediction_fragment(fragment)
            yield fragment

    def result(self) -> FakeResult:
        return FakeResult("".join(self.fragments))


class FakeModel:
    """Stand-in for an LM Studio model handle exposing ``respond``."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def respond(self, history, *, response_format=None, **kwargs):
        self.calls.append({"history": history, "response_format": response_format})
        return FakeResult(self.content)


class FakeStreamingModel(FakeModel):
    def __init__(self, fragments: list[str]) -> None:
        super().__init__("".join(fragments))
        self.fragments = fragments

    def respond_stream(self, history, *, response_format=None, on_prediction_fragment=None, **kwargs):
        self.calls.append(
            {
                "history": history,
                "response_format": response_format,
                "on_prediction_fragment": on_prediction_fragment,
            }
        )
        return FakeStream(self.fragments, on_prediction_fragment)


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


def test_provider_streams_structured_outline_fragments(tmp_path: Path) -> None:
    model = FakeStreamingModel(
        [
            '{"title":"Episode","segments":[',
            '{"title":"Intro","summary":"Set context","target_minutes":2,"source_focus":["notes"]}',
            "]}",
        ]
    )
    provider = LMStudioProvider(model=model)
    parsed = ParsedSources(markdown="# Notes\n", sources=[])
    fragments: list[str] = []

    outline = provider.make_outline(parsed, make_request(tmp_path), text_progress=fragments.append)

    assert outline.title == "Episode"
    assert fragments == model.fragments
    assert model.calls[0]["on_prediction_fragment"] is not None


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


def test_provider_parses_script_delivery_instructions(tmp_path: Path) -> None:
    model = FakeModel(
        '{"turns":[{"speaker":"Host A","text":"That changes the stakes.",'
        '"delivery_instruction":"slow and serious, with restrained tension"}]}'
    )
    provider = LMStudioProvider(model=model)
    segment = OutlineSegment(
        title="Risk",
        summary="A serious consequence appears",
        target_minutes=2,
        source_focus=["notes"],
    )

    turns = provider.make_script_segment(segment, "", make_request(tmp_path))

    assert turns[0].text == "That changes the stakes."
    assert turns[0].delivery_instruction == "slow and serious, with restrained tension"


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
    assert "delivery_instruction" in turns_schema["items"]["properties"]
    assert "delivery_instruction" in turns_schema["items"]["required"]


def test_provider_script_prompt_requests_delivery_instructions(tmp_path: Path) -> None:
    model = FakeModel('{"turns":[{"speaker":"Host A","text":"Welcome."}]}')
    provider = LMStudioProvider(model=model)
    segment = OutlineSegment(
        title="Intro",
        summary="Open with a joke before a serious claim",
        target_minutes=2,
        source_focus=["notes"],
    )

    provider.make_script_segment(segment, "", make_request(tmp_path))

    prompt = _last_user_prompt(model)
    assert "delivery_instruction" in prompt
    assert "Do not include delivery notes in text" in prompt
    assert "jokes, serious claims, sad moments, tension, pacing" in prompt


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

    # No host -> use the app's documented LM Studio default; no model -> reuse the loaded model.
    assert created == {"host": DEFAULT_LM_STUDIO_HOST, "model_key": None}


def test_list_available_models_uses_documented_default_host(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict = {}

    class FakeLlmNamespace:
        def list_downloaded(self):
            return []

    class FakeClient:
        def __init__(self, host=None) -> None:
            created["host"] = host
            self.llm = FakeLlmNamespace()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm.lms, "Client", FakeClient)

    assert list_available_models() == []
    assert created == {"host": DEFAULT_LM_STUDIO_HOST}


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
