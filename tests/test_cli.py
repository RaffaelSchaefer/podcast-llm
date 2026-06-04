import json
from pathlib import Path

from podcast_llm import cli
from podcast_llm.cli import load_request_from_config
from podcast_llm.models import GenerationResult


def test_load_request_from_config(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "source_paths": [str(source)],
                "language": "de",
                "duration_minutes": 12,
                "export_format": "mp3",
                "lmstudio_host": "localhost:1234",
                "custom_instructions": "Make the tone practical and skeptical.",
            }
        ),
        encoding="utf-8",
    )

    request = load_request_from_config(config)

    assert request.language == "de"
    assert request.duration_minutes == 12
    assert request.export_format == "mp3"
    assert request.lmstudio_host == "localhost:1234"
    assert request.custom_instructions == "Make the tone practical and skeptical."


def test_main_generate_uses_config_and_pipeline(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_paths": [str(source)], "language": "en", "duration_minutes": 3}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs" / "episode"
    transcript = output_dir / "transcript.md"
    audio = output_dir / "episode.wav"

    def fake_run_generate(request):
        assert request.source_paths == [source]
        return GenerationResult(output_dir=output_dir, transcript_path=transcript, audio_path=audio)

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_generate", fake_run_generate)

    cli.main(["generate", "--config", str(config)])

    output = capsys.readouterr().out
    assert f"Transcript: {transcript}" in output
    assert f"Audio: {audio}" in output


def test_main_generate_reports_mp3_as_single_audio_output(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_paths": [str(source)], "language": "en", "duration_minutes": 3}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs" / "episode"
    transcript = output_dir / "metadata" / "transcript.md"
    mp3 = output_dir / "episode.mp3"

    def fake_run_generate(request):
        assert request.source_paths == [source]
        return GenerationResult(output_dir=output_dir, transcript_path=transcript, audio_path=mp3, mp3_path=mp3)

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_generate", fake_run_generate)

    cli.main(["generate", "--config", str(config)])

    output = capsys.readouterr().out
    assert f"Transcript: {transcript}" in output
    assert f"Audio: {mp3}" in output
    assert "MP3:" not in output
