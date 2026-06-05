from pathlib import Path

import pytest
from textual.widgets import Button, Checkbox, Input, ProgressBar, RadioButton, Select, Static, TextArea

from podcast_llm.models import GenerationResult
from podcast_llm.presets import preset_by_id
from podcast_llm.progress import ProgressEvent
from podcast_llm.tui import PodcastWizard
from podcast_llm.tui.draft import PodcastDraft
from podcast_llm.tui.screens.generation import (
    GenerationDone,
    GenerationFailed,
    GenerationScreen,
    ProgressUpdate,
)
from podcast_llm.tui.screens.review import ReviewScreen
from podcast_llm.tui.screens.sources import SourcesScreen
from podcast_llm.tui.screens.style import StyleScreen
from podcast_llm.tui.screens.voices import VoicesScreen
from podcast_llm.tui.widgets import GENERATION_PHASES, PhaseChecklist


def _write_source(directory: Path, name: str) -> Path:
    source = directory / name
    source.write_text("# Notes\n", encoding="utf-8")
    return source


@pytest.mark.anyio
async def test_wizard_builds_generation_request_from_all_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path, "notes.md")
    monkeypatch.setenv("PODCAST_LLM_HOST", "100.124.83.79:1234")
    monkeypatch.setenv("PODCAST_LLM_MODEL", "qwen/qwen3.6-35b-a3b")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SourcesScreen)
        app.screen.add_source(source)
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        style = app.screen
        assert isinstance(style, StyleScreen)
        style.query_one("#language", Select).value = "de"
        style.query_one("#duration", Input).value = "12"
        style.query_one("#custom_instructions", TextArea).text = "Keep it lively but technically precise."
        style.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        voices.query_one("#host_a", Select).value = "Aiden"
        voices.query_one("#host_b", Select).value = "Serena"
        voices.query_one("#tts_mode", Select).value = "apple"
        voices.query_one("#export", Select).value = "mp3"
        voices.query_one("#background_music", Checkbox).value = True
        voices.action_next()
        await pilot.pause()

        assert isinstance(app.screen, ReviewScreen)
        request = app.draft.to_request()

    assert request.source_paths == [source]
    assert request.language == "de"
    assert request.duration_minutes == 12
    assert request.host_a_voice == "Aiden"
    assert request.host_b_voice == "Serena"
    assert request.tts_mode == "apple"
    assert request.lmstudio_host == "100.124.83.79:1234"
    assert request.lmstudio_model == "qwen/qwen3.6-35b-a3b"
    assert request.export_format == "mp3"
    assert request.custom_instructions == "Keep it lively but technically precise."
    assert request.enable_background_music is True


@pytest.mark.anyio
async def test_voices_screen_prefills_env_model_and_overrides_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path, "notes.md")
    monkeypatch.setenv("PODCAST_LLM_HOST", "100.124.83.79:1234")
    monkeypatch.setenv("PODCAST_LLM_MODEL", "qwen/qwen3.6-35b-a3b")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        assert list(voices.query("#openai_api_key")) == []
        assert voices.query_one("#lmstudio_model", Select).value == "qwen/qwen3.6-35b-a3b"
        voices.query_one("#lmstudio_host", Input).value = "localhost:4321"
        voices.commit(strict=True)

    assert app.draft.lmstudio_host == "localhost:4321"
    assert app.draft.lmstudio_model == "qwen/qwen3.6-35b-a3b"
    assert app.draft.tts_mode == "auto"
    request = app.draft.to_request()
    assert request.lmstudio_host == "localhost:4321"
    assert request.lmstudio_model == "qwen/qwen3.6-35b-a3b"
    assert request.tts_mode == "auto"


@pytest.mark.anyio
async def test_voices_connection_test_uses_env_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path, "notes.md")
    monkeypatch.setenv("PODCAST_LLM_HOST", "100.124.83.79:1234")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        captured: dict[str, str | None] = {}

        def fake_test_connection(host: str | None) -> None:
            captured["host"] = host

        voices._test_connection = fake_test_connection  # type: ignore[method-assign]
        voices.query_one("#test_connection", Button).press()
        await pilot.pause()

    assert captured == {"host": "100.124.83.79:1234"}


@pytest.mark.anyio
async def test_voices_connection_test_populates_model_selector(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        voices._show_connection_models(["llama-3", "phi-3"])
        await pilot.pause()
        model_select = voices.query_one("#lmstudio_model", Select)
        model_select.value = "phi-3"
        voices.commit(strict=True)

    assert app.draft.lmstudio_model == "phi-3"


@pytest.mark.anyio
async def test_voices_auto_loads_models_on_mount(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        captured: dict[str, str | None] = {}

        def fake_test_connection(host: str | None) -> None:
            captured["host"] = host

        voices._test_connection = fake_test_connection  # type: ignore[method-assign]
        voices.on_mount()

    assert captured == {"host": None}


@pytest.mark.anyio
async def test_voices_connection_failure_does_not_block_next(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        voices._show_connection_error(RuntimeError("LM Studio is not running"))
        voices.action_next()
        await pilot.pause()

        assert isinstance(app.screen, ReviewScreen)


@pytest.mark.anyio
async def test_selected_model_survives_host_change(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        voices = app.screen
        assert isinstance(voices, VoicesScreen)
        voices._show_connection_models(["llama-3", "phi-3"])
        await pilot.pause()
        voices.query_one("#lmstudio_model", Select).value = "llama-3"
        voices.query_one("#lmstudio_host", Input).value = "localhost:4321"
        voices.commit(strict=True)

    assert app.draft.lmstudio_host == "localhost:4321"
    assert app.draft.lmstudio_model == "llama-3"


@pytest.mark.anyio
async def test_sources_screen_collects_added_file(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        await pilot.pause()

    assert app.draft.source_paths == [source]


@pytest.mark.anyio
async def test_sources_screen_deduplicates_added_and_manual_files(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    other_source = _write_source(tmp_path, "other.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.add_source(source)
        add_path = screen.query_one("#add_path", Input)
        add_path.focus()
        add_path.value = f"{source}; {other_source}; {source}"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert app.draft.source_paths == [source, other_source]


@pytest.mark.anyio
async def test_sources_toggle_removes_file(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.toggle_source(source)
        assert app.draft.source_paths == [source]
        screen.toggle_source(source)
        assert app.draft.source_paths == []


@pytest.mark.anyio
async def test_sources_add_all_in_folder_adds_supported_only(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    (tmp_path / "sub").mkdir()
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.navigate_to(tmp_path)
        await pilot.pause()
        screen.add_all_in_current_dir()
        await pilot.pause()

    assert sorted(p.name for p in app.draft.source_paths) == ["a.md", "b.txt"]


@pytest.mark.anyio
async def test_sources_add_by_glob(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.navigate_to(tmp_path)
        await pilot.pause()
        screen.add_by_pattern("*.md")
        await pilot.pause()

    assert sorted(p.name for p in app.draft.source_paths) == ["a.md", "b.md"]


@pytest.mark.anyio
async def test_sources_navigate_to_missing_folder_is_safe(tmp_path: Path) -> None:
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        before = screen._current_dir
        screen.navigate_to(tmp_path / "does-not-exist")
        await pilot.pause()
        assert screen._current_dir == before


@pytest.mark.anyio
async def test_sources_screen_rejects_empty_selection() -> None:
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        with pytest.raises(ValueError, match="Select at least one source file"):
            app.screen.commit(strict=True)
        app.screen.action_next()
        await pilot.pause()
        assert isinstance(app.screen, SourcesScreen)  # blocked, did not advance


@pytest.mark.anyio
async def test_style_preset_prefills_duration_and_instructions(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    deep_dive = preset_by_id("deep_dive")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()

        style = app.screen
        assert isinstance(style, StyleScreen)
        style.query_one("#preset_deep_dive", RadioButton).value = True
        await pilot.pause()
        assert style.query_one("#duration", Input).value == str(deep_dive.duration_minutes)
        assert style.query_one("#custom_instructions", TextArea).text == deep_dive.custom_instructions
        style.action_next()
        await pilot.pause()

    assert app.draft.preset_id == "deep_dive"
    assert app.draft.duration_minutes == deep_dive.duration_minutes


def test_draft_to_request_uses_env_for_llm_endpoint_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "notes.md"
    monkeypatch.setenv("PODCAST_LLM_HOST", "100.124.83.79:1234")
    monkeypatch.setenv("PODCAST_LLM_MODEL", "qwen/qwen3.6-35b-a3b")
    draft = PodcastDraft(
        source_paths=[source],
        language="de",
        duration_minutes=20,
        host_a_voice="M2",
        host_b_voice="F3",
        custom_instructions="Be concise.",
        export_format="mp3",
    )

    request = draft.to_request()

    assert request.source_paths == [source]
    assert request.language == "de"
    assert request.duration_minutes == 20
    assert request.host_a_voice == "M2"
    assert request.host_b_voice == "F3"
    assert request.tts_mode == "auto"
    assert request.custom_instructions == "Be concise."
    assert request.lmstudio_host == "100.124.83.79:1234"
    assert request.lmstudio_model == "qwen/qwen3.6-35b-a3b"
    assert request.export_format == "mp3"


@pytest.mark.anyio
async def test_review_summary_shows_env_llm_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path, "notes.md")
    monkeypatch.setenv("PODCAST_LLM_HOST", "100.124.83.79:1234")
    monkeypatch.setenv("PODCAST_LLM_MODEL", "qwen/qwen3.6-35b-a3b")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        summary = screen._summary()

    assert "| LLM endpoint | 100.124.83.79:1234 |" in summary
    assert "| Model | qwen/qwen3.6-35b-a3b |" in summary
    assert "| TTS mode | auto |" in summary
    assert "| TTS model | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |" in summary
    assert "| Quiet dynamic background music | No |" in summary


@pytest.mark.anyio
async def test_review_summary_shows_enabled_background_music(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.draft.enable_background_music = True
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()
        app.screen.action_next()
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        summary = screen._summary()

    assert "| Quiet dynamic background music | Yes |" in summary


@pytest.mark.anyio
async def test_generation_screen_updates_progress(tmp_path: Path) -> None:
    request = PodcastDraft(source_paths=[tmp_path / "notes.md"], language="en", duration_minutes=2).to_request()
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = GenerationScreen(request, auto_start=False)
        app.push_screen(screen)
        await pilot.pause()

        bar = screen.query_one("#gen_bar", ProgressBar)
        assert bar.total is None  # indeterminate until turn totals are known

        screen.on_progress_update(ProgressUpdate(ProgressEvent("parse", "Parsing…")))
        screen.on_progress_update(ProgressUpdate(ProgressEvent("synthesize", "turn 1", 1, 4)))
        await pilot.pause()

        assert bar.total == 4
        assert bar.progress == 1
        checklist = screen.query_one("#phases", PhaseChecklist)
        assert checklist._max_index == GENERATION_PHASES.index("synthesize")

        screen.on_generation_done(
            GenerationDone(
                GenerationResult(
                    output_dir=tmp_path / "out",
                    transcript_path=tmp_path / "out" / "metadata" / "transcript.md",
                    audio_path=tmp_path / "out" / "episode.wav",
                )
            )
        )
        await pilot.pause()

        assert checklist._max_index == len(GENERATION_PHASES)
        assert screen.query_one("#gen_open", Button).disabled is False
        assert screen.query_one("#gen_restart", Button).disabled is False


@pytest.mark.anyio
async def test_generation_screen_keeps_bar_indeterminate_off_synthesize(tmp_path: Path) -> None:
    request = PodcastDraft(source_paths=[tmp_path / "notes.md"], language="en", duration_minutes=2).to_request()
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = GenerationScreen(request, auto_start=False)
        app.push_screen(screen)
        await pilot.pause()

        bar = screen.query_one("#gen_bar", ProgressBar)
        screen.on_progress_update(ProgressUpdate(ProgressEvent("synthesize", "turn 1", 1, 4)))
        await pilot.pause()
        assert bar.total == 4

        # Moving on to a phase without a real total should not leave the bar stuck.
        screen.on_progress_update(ProgressUpdate(ProgressEvent("export", "writing wav")))
        await pilot.pause()
        assert bar.total is None


@pytest.mark.anyio
async def test_generation_screen_renders_live_script_fragments(tmp_path: Path) -> None:
    request = PodcastDraft(source_paths=[tmp_path / "notes.md"], language="en", duration_minutes=2).to_request()
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = GenerationScreen(request, auto_start=False)
        app.push_screen(screen)
        await pilot.pause()

        screen.on_progress_update(
            ProgressUpdate(
                ProgressEvent(
                    "script",
                    kind="text_reset",
                    text_scope="script",
                    seg=1,
                    segment_title="Intro",
                )
            )
        )
        screen.on_progress_update(
            ProgressUpdate(
                ProgressEvent(
                    "script",
                    kind="text_fragment",
                    text='{"turns":[{"speaker":"Host A","text":"Welcome to this"',
                    text_scope="script",
                    seg=1,
                    segment_title="Intro",
                )
            )
        )
        await pilot.pause()

        live_text = screen.query_one("#live_text", Static)
        assert "Host A" in str(live_text.content) and "Welcome to this" in str(live_text.content)

        screen.on_progress_update(
            ProgressUpdate(
                ProgressEvent(
                    "script",
                    kind="text_replace",
                    text="Host A: Welcome to this episode.",
                    text_scope="script",
                    seg=1,
                    segment_title="Intro",
                )
            )
        )
        await pilot.pause()

        assert "Host A" in str(live_text.content) and "Welcome to this episode." in str(live_text.content)


@pytest.mark.anyio
async def test_generation_failure_enables_restart_only(tmp_path: Path) -> None:
    request = PodcastDraft(source_paths=[tmp_path / "notes.md"], language="en", duration_minutes=2).to_request()
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = GenerationScreen(request, auto_start=False)
        app.push_screen(screen)
        await pilot.pause()

        screen.on_generation_failed(GenerationFailed(RuntimeError("boom")))
        await pilot.pause()

        assert screen.query_one("#gen_restart", Button).disabled is False
        assert screen.query_one("#gen_open", Button).disabled is True


@pytest.mark.anyio
async def test_generation_screen_reports_qwen_preflight_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = PodcastDraft(source_paths=[tmp_path / "notes.md"], language="en", duration_minutes=2).to_request()
    app = PodcastWizard()

    def fail_preflight() -> None:
        raise RuntimeError("Qwen TTS startup failed: bad value(s) in fds_to_keep")

    monkeypatch.setattr("podcast_llm.tui.screens.generation.preflight_qwen_runtime", fail_preflight)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = GenerationScreen(request)
        screen._run = lambda: (_ for _ in ()).throw(AssertionError("generation worker should not start"))
        app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#gen_restart", Button).disabled is False
        assert screen.query_one("#gen_open", Button).disabled is True
        assert "Qwen TTS startup failed" in str(screen.query_one("#gen_title", Static).content)


@pytest.mark.anyio
async def test_rail_goto_step_allows_visited_and_blocks_future(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()  # -> Style (visits index 1)
        await pilot.pause()
        assert isinstance(app.screen, StyleScreen)

        # Jumping forward to an unvisited step is rejected.
        app.goto_step(3)
        await pilot.pause()
        assert isinstance(app.screen, StyleScreen)

        # Jumping back to a visited step works.
        app.goto_step(0)
        await pilot.pause()
        assert isinstance(app.screen, SourcesScreen)


@pytest.mark.anyio
async def test_restart_resets_draft_and_returns_to_sources(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "notes.md")
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.add_source(source)
        app.screen.action_next()
        await pilot.pause()
        assert isinstance(app.screen, StyleScreen)

        app.restart()
        await pilot.pause()

        assert isinstance(app.screen, SourcesScreen)
        assert app.draft.source_paths == []
        assert app._step_index == 0
        assert app._max_index == 0
