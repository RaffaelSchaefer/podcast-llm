import json
from typing import Protocol

import lmstudio as lms
from pydantic import ValidationError

from .models import DialogueTurn, EpisodeOutline, GenerationRequest, OutlineSegment, ParsedSources


class LLMOutputError(RuntimeError):
    pass


# LM Studio's local server speaks on this host:port by default. Note the SDK
# wants a bare ``host:port`` (it builds the ``http://``/``ws://`` URLs itself),
# unlike the OpenAI-compatible REST endpoint which lives under ``/v1``.
DEFAULT_LM_STUDIO_HOST = "localhost:1234"


class LLMProvider(Protocol):
    def make_outline(self, parsed_sources: ParsedSources, request: GenerationRequest) -> EpisodeOutline:
        ...

    def make_script_segment(
        self,
        outline_segment: OutlineSegment,
        prior_context: str,
        request: GenerationRequest,
    ) -> list[DialogueTurn]:
        ...


class LMStudioProvider:
    """Drive a locally loaded LM Studio model through the ``lmstudio`` SDK.

    ``model`` lets tests inject a stand-in handle exposing ``.respond``. In
    production we resolve a handle from the LM Studio server: ``model_key``
    loads (or reuses) a named model, and a ``None`` key reuses whichever model
    is already loaded.
    """

    def __init__(
        self,
        model: object | None = None,
        *,
        model_key: str | None = None,
        host: str | None = None,
    ) -> None:
        if model is not None:
            self.model = model
            return
        self.model = _resolve_model(model_key, host)

    def make_outline(self, parsed_sources: ParsedSources, request: GenerationRequest) -> EpisodeOutline:
        custom_instructions = _custom_instructions_prompt(request)
        content = self._chat_json(
            schema=_episode_outline_schema(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You design concise, engaging two-host podcast outlines from source material. "
                        "Keep the output grounded in the supplied sources."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return JSON with this shape: "
                        '{"title": str, "segments": [{"title": str, "summary": str, '
                        '"target_minutes": int, "source_focus": [str]}]}. '
                        f"Target language: {request.language}. "
                        f"Target total duration: {request.duration_minutes} minutes. "
                        "Create enough segments for reliable chunked scripting.\n\n"
                        f"{custom_instructions}"
                        f"Sources:\n{parsed_sources.markdown}"
                    ),
                },
            ],
        )
        try:
            return EpisodeOutline.model_validate_json(content)
        except ValidationError as exc:
            raise LLMOutputError(f"Invalid outline JSON: {exc}") from exc

    def make_script_segment(
        self,
        outline_segment: OutlineSegment,
        prior_context: str,
        request: GenerationRequest,
    ) -> list[DialogueTurn]:
        custom_instructions = _custom_instructions_prompt(request)
        content = self._chat_json(
            schema=_dialogue_turns_schema(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write natural NotebookLM-style two-host explainer dialogue. "
                        "Use only Host A and Host B. Keep each turn suitable for text-to-speech."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return JSON with this shape: "
                        '{"turns": [{"speaker": "Host A"|"Host B", "text": str}]}. '
                        f"Target language: {request.language}. "
                        f"Segment title: {outline_segment.title}. "
                        f"Segment summary: {outline_segment.summary}. "
                        f"Target length: {outline_segment.target_minutes} minutes. "
                        f"Source focus: {', '.join(outline_segment.source_focus)}. "
                        f"{custom_instructions}"
                        f"Prior context for continuity: {prior_context or 'None'}"
                    ),
                },
            ],
        )
        try:
            payload = json.loads(content)
            turns = payload.get("turns", payload)
            return [DialogueTurn.model_validate(turn) for turn in turns]
        except (TypeError, ValueError, ValidationError) as exc:
            raise LLMOutputError(f"Invalid script JSON: {exc}") from exc

    def _chat_json(self, schema: dict, messages: list[dict[str, str]]) -> str:
        # The SDK accepts this history dict directly (it runs it through
        # ``Chat.from_history`` internally), so we don't build a Chat ourselves.
        history = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
        result = self.model.respond(
            history,
            response_format={"type": "json", "jsonSchema": schema},
        )
        content = result.content
        if not content:
            raise LLMOutputError("LM Studio returned an empty response")
        return content


def provider_for_request(request: GenerationRequest) -> LLMProvider:
    if request.llm_provider == "lmstudio":
        return LMStudioProvider(model_key=request.lmstudio_model, host=request.lmstudio_host)
    raise ValueError(f"Unsupported LLM provider: {request.llm_provider}")


def list_available_models(host: str | None = None) -> list[str]:
    """Ask an LM Studio server which LLMs it can serve.

    Used by the TUI to confirm a connection and to offer the downloaded model
    keys. Raises whatever the ``lmstudio`` SDK raises on a failed connection so
    callers can surface the reason.
    """
    with _client(host) as client:
        return sorted(model.model_key for model in client.llm.list_downloaded())


def _resolve_model(model_key: str | None, host: str | None) -> object:
    """Resolve an LM Studio model handle, loading the named model if needed."""
    client = _client(host)
    if model_key:
        return client.llm.model(model_key)
    return client.llm.model()


def _client(host: str | None) -> lms.Client:
    """Open an LM Studio client, honouring a custom ``host:port`` when given."""
    return lms.Client(host) if host else lms.Client()


def _custom_instructions_prompt(request: GenerationRequest) -> str:
    instructions = request.custom_instructions.strip()
    if not instructions:
        return ""
    return f"Custom podcast instructions:\n{instructions}\n\n"


def _episode_outline_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "target_minutes": {"type": "integer", "minimum": 1},
                        "source_focus": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["title", "summary", "target_minutes", "source_focus"],
                },
            },
        },
        "required": ["title", "segments"],
    }


def _dialogue_turns_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "speaker": {"type": "string", "enum": ["Host A", "Host B"]},
                        "text": {"type": "string", "minLength": 1},
                    },
                    "required": ["speaker", "text"],
                },
            },
        },
        "required": ["turns"],
    }
