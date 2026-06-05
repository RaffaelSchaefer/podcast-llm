from pathlib import Path

import numpy as np
import soundfile as sf

from podcast_llm.models import (
    DialogueTurn,
    EpisodeOutline,
    GenerationRequest,
    OutlineSegment,
    ParsedSource,
    ParsedSources,
)
from podcast_llm.pipeline import PodcastPipeline


class FakeParser:
    def parse(self, paths, request=None):
        return ParsedSources(
            markdown="# Source\nUseful facts",
            sources=[ParsedSource(path=Path(paths[0]), markdown="# Source\nUseful facts")],
        )


class FakeProvider:
    def make_outline(self, parsed_sources, request, text_progress=None):
        if text_progress is not None:
            text_progress('{"title":"Test Episode",')
            text_progress('"segments":[{"title":"Part One"')
        return EpisodeOutline(
            title="Test Episode",
            segments=[
                OutlineSegment(
                    title="Part One",
                    summary="First idea",
                    target_minutes=1,
                    source_focus=["source"],
                ),
                OutlineSegment(
                    title="Part Two",
                    summary="Second idea",
                    target_minutes=1,
                    source_focus=["source"],
                ),
            ],
        )

    def make_script_segment(self, outline_segment, prior_context, request, text_progress=None):
        if text_progress is not None:
            text_progress('{"turns":[{"speaker":"Host A",')
            text_progress(f'"text":"{outline_segment.title} from A"')
        return [
            DialogueTurn(speaker="Host A", text=f"{outline_segment.title} from A"),
            DialogueTurn(speaker="Host B", text=f"{outline_segment.title} from B"),
        ]


class FakeSynthesizer:
    sample_rate = 44_100

    def synthesize_turn(self, turn, request):
        return np.zeros((10,), dtype=np.float32)


class FakeMusicGenerator:
    model_id = "fake/music"

    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, duration_seconds: float, sample_rate: int) -> np.ndarray:
        self.calls.append(
            {
                "prompt": prompt,
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
            }
        )
        return np.ones((max(1, round(duration_seconds * sample_rate)),), dtype=np.float32)


def test_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        output_dir=tmp_path / "outputs",
    )
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
    )

    result = pipeline.generate(request)

    metadata_dir = result.output_dir / "metadata"
    assert result.output_dir.name == "test-episode"
    assert (metadata_dir / "sources.md").read_text(encoding="utf-8") == "# Source\nUseful facts"
    assert "## Part One" in (metadata_dir / "transcript.md").read_text(encoding="utf-8")
    assert (metadata_dir / "outline.json").exists()
    assert (metadata_dir / "run.json").exists()
    assert not (result.output_dir / "sources.md").exists()
    assert not (result.output_dir / "outline.json").exists()
    assert not (result.output_dir / "transcript.md").exists()
    assert not (result.output_dir / "run.json").exists()
    assert not (result.output_dir / "segments").exists()
    assert result.transcript_path == metadata_dir / "transcript.md"
    assert result.audio_path == result.output_dir / "episode.wav"
    assert (result.output_dir / "episode.wav").exists()

    audio, sample_rate = sf.read(result.output_dir / "episode.wav", dtype="float32")
    expected_seams = 3 * round(FakeSynthesizer.sample_rate * 0.25)
    assert sample_rate == FakeSynthesizer.sample_rate
    assert len(audio) == 4 * 10 + expected_seams


def test_pipeline_skips_background_music_when_disabled(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        output_dir=tmp_path / "outputs",
    )
    music_generator = FakeMusicGenerator()
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
        music_generator=music_generator,
    )

    result = pipeline.generate(request)

    assert music_generator.calls == []
    assert not (result.output_dir / "metadata" / "background_music.json").exists()


def test_pipeline_generates_section_background_music_metadata(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        enable_background_music=True,
        output_dir=tmp_path / "outputs",
    )
    music_generator = FakeMusicGenerator()
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
        music_generator=music_generator,
    )

    result = pipeline.generate(request)

    assert len(music_generator.calls) == 2
    assert all("quiet instrumental background music" in call["prompt"] for call in music_generator.calls)
    metadata = (result.output_dir / "metadata" / "background_music.json").read_text(encoding="utf-8")
    assert '"model_id": "fake/music"' in metadata
    assert '"title": "Part One"' in metadata
    assert '"title": "Part Two"' in metadata
    audio, _sample_rate = sf.read(result.output_dir / "episode.wav", dtype="float32")
    assert np.max(audio) > 0


def test_pipeline_fails_when_enabled_background_music_generation_fails(tmp_path: Path) -> None:
    class FailingMusicGenerator(FakeMusicGenerator):
        def generate(self, prompt: str, duration_seconds: float, sample_rate: int) -> np.ndarray:
            raise RuntimeError("ACE-Step failed")

    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        enable_background_music=True,
        output_dir=tmp_path / "outputs",
    )
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
        music_generator=FailingMusicGenerator(),
    )

    try:
        pipeline.generate(request)
    except RuntimeError as exc:
        assert "ACE-Step failed" in str(exc)
    else:
        raise AssertionError("Expected background music failure to propagate")


def test_pipeline_fails_when_enabled_background_music_is_silent(tmp_path: Path) -> None:
    class SilentMusicGenerator(FakeMusicGenerator):
        def generate(self, prompt: str, duration_seconds: float, sample_rate: int) -> np.ndarray:
            return np.zeros((max(1, round(duration_seconds * sample_rate)),), dtype=np.float32)

    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        enable_background_music=True,
        output_dir=tmp_path / "outputs",
    )
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
        music_generator=SilentMusicGenerator(),
    )

    try:
        pipeline.generate(request)
    except RuntimeError as exc:
        assert "silent background music" in str(exc)
    else:
        raise AssertionError("Expected silent background music failure to propagate")


def test_pipeline_deletes_intermediate_wav_after_mp3_export(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        export_format="mp3",
        output_dir=tmp_path / "outputs",
    )
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
    )

    def fake_export_mp3(wav_path: Path, mp3_path: Path, request: GenerationRequest) -> Path:
        assert wav_path.exists()
        mp3_path.write_bytes(b"fake mp3")
        return mp3_path

    monkeypatch.setattr("podcast_llm.pipeline.maybe_export_mp3", fake_export_mp3)

    result = pipeline.generate(request)

    assert result.mp3_path == result.output_dir / "episode.mp3"
    assert result.audio_path == result.mp3_path
    assert result.mp3_path.exists()
    assert not (result.output_dir / "episode.wav").exists()
    assert not (result.output_dir / "segments").exists()


def test_pipeline_emits_live_text_generation_events(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    request = GenerationRequest(
        source_paths=[source],
        language="en",
        duration_minutes=2,
        output_dir=tmp_path / "outputs",
    )
    pipeline = PodcastPipeline(
        parser=FakeParser(),
        llm_provider=FakeProvider(),
        synthesizer=FakeSynthesizer(),
    )
    events = []

    pipeline.generate(request, events.append)

    text_events = [event for event in events if event.kind.startswith("text_")]
    assert [event.kind for event in text_events[:2]] == ["text_reset", "text_fragment"]
    assert text_events[0].text_scope == "outline"
    assert any(event.text_scope == "script" and event.kind == "text_fragment" for event in text_events)
    assert any(
        event.kind == "text_replace" and "Host A: Part One from A" in event.text
        for event in text_events
    )
