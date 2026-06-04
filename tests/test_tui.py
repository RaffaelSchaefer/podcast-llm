from pathlib import Path

import pytest
from textual.widgets import Button, Input, ProgressBar, RadioButton, Select, Static, TextArea

from podcast_llm.models import GenerationResult
from podcast_llm.presets import preset_by_id
from podcast_llm.progress import ProgressEvent
from podcast_llm.tui import PodcastWizard
from podcast_llm.tui.draft import PodcastDraft
from podcast_llm.tui.screens.generation import GenerationDone, GenerationScreen, ProgressUpdate
from podcast_llm.tui.screens.review import ReviewScreen
from podcast_llm.tui.screens.sources import SourcesScreen
from podcast_llm.tui.screens.style import StyleScreen
from podcast_llm.tui.screens.voices import VoicesScreen
from podcast_llm.tui.ascii_art import CANVAS_H, CANVAS_W, FRAMES
from podcast_llm.tui.widgets import GENERATION_PHASES, PhaseAnimation, PhaseChecklist


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
        voices.query_one("#host_a", Select).value = "M1"
        voices.query_one("#host_b", Select).value = "F1"
        voices.query_one("#export", Select).value = "mp3"
        voices.action_next()
        await pilot.pause()

        assert isinstance(app.screen, ReviewScreen)
        request = app.draft.to_request()

    assert request.source_paths == [source]
    assert request.language == "de"
    assert request.duration_minutes == 12
    assert request.host_a_voice == "M1"
    assert request.host_b_voice == "F1"
    assert request.lmstudio_host == "100.124.83.79:1234"
    assert request.lmstudio_model == "qwen/qwen3.6-35b-a3b"
    assert request.export_format == "mp3"
    assert request.custom_instructions == "Keep it lively but technically precise."


@pytest.mark.anyio
async def test_voices_screen_leaves_model_to_env_and_overrides_host(
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
        # The model has no input field — it always comes from the environment.
        assert list(voices.query("#lmstudio_model")) == []
        assert list(voices.query("#openai_api_key")) == []
        voices.query_one("#lmstudio_host", Input).value = "localhost:4321"
        voices.commit(strict=True)

    assert app.draft.lmstudio_host == "localhost:4321"
    request = app.draft.to_request()
    assert request.lmstudio_host == "localhost:4321"
    assert request.lmstudio_model == "qwen/qwen3.6-35b-a3b"


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
async def test_voices_connection_test_reports_models_without_model_input(tmp_path: Path) -> None:
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
        assert list(voices.query("#lmstudio_model")) == []


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


def test_ascii_frames_share_a_fixed_canvas() -> None:
    from rich.text import Text

    for phase, frames in FRAMES.items():
        assert frames, f"{phase} has no frames"
        for frame in frames:
            lines = frame.split("\n")
            assert len(lines) == CANVAS_H, f"{phase} frame has {len(lines)} rows"
            for line in lines:
                width = len(Text.from_markup(line).plain)
                assert width == CANVAS_W, f"{phase} line width {width}: {line!r}"


def test_ascii_frames_are_theater_scale_and_do_not_reuse_old_art() -> None:
    art = "\n".join(frame for frames in FRAMES.values() for frame in frames)
    old_animation_markers = (
        "Parsing the sources",
        "Drafting the outline",
        "Writing the script",
        "QWERTYUIOP",
        "FAC-TORY",
    )

    assert CANVAS_W >= 60
    assert CANVAS_H >= 15
    for marker in old_animation_markers:
        assert marker not in art


def test_phase_animation_set_phase_selects_frames_and_ignores_no_ops() -> None:
    anim = PhaseAnimation()
    assert anim._frames is FRAMES["parse"]

    anim.set_phase("export")
    assert anim._frames is FRAMES["export"]

    anim._index = 2
    anim.set_phase("export")  # repeat of current phase is a no-op
    assert anim._index == 2

    anim.set_phase("nonexistent")  # unknown phase is a no-op
    assert anim._frames is FRAMES["export"]
    assert anim._index == 2

    anim.set_phase("done")
    assert anim._frames is FRAMES["done"]
    assert anim._index == 0


def test_phase_animation_selects_synthesize_frames_for_active_speaker() -> None:
    anim = PhaseAnimation()

    anim.set_phase("synthesize", speaker="A")
    host_a_frame = anim._frames[0]

    anim.set_phase("synthesize", speaker="B")
    host_b_frame = anim._frames[0]

    assert host_a_frame != host_b_frame
    assert "HOST A" in host_a_frame
    assert "HOST B" in host_b_frame


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
        assert screen.query_one("#gen_back", Button).disabled is False


@pytest.mark.anyio
async def test_generation_screen_defaults_to_theater_and_toggles_log_view(tmp_path: Path) -> None:
    request = PodcastDraft(source_paths=[tmp_path / "notes.md"], language="en", duration_minutes=2).to_request()
    app = PodcastWizard()

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = GenerationScreen(request, auto_start=False)
        app.push_screen(screen)
        await pilot.pause()

        theater = screen.query_one("#theater_panel")
        log_panel = screen.query_one("#log_panel")
        hint = screen.query_one("#view_hint", Static)

        assert theater.display is True
        assert log_panel.display is False
        assert "L" in str(hint.content)

        await pilot.press("l")
        await pilot.pause()

        assert theater.display is False
        assert log_panel.display is True
        assert "theater" in str(hint.content).lower()

        await pilot.press("l")
        await pilot.pause()

        assert theater.display is True
        assert log_panel.display is False
