import numpy as np

from podcast_llm.export import concat_audio, maybe_export_mp3, write_wav
from podcast_llm.models import GenerationRequest


def test_concat_audio_inserts_silence_between_chunks() -> None:
    audio = concat_audio(
        [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
            np.array([0.5, 0.6], dtype=np.float32),
        ],
        sample_rate=10,
        seam_silence_seconds=0.2,
    )

    np.testing.assert_array_equal(
        audio,
        np.array([0.1, 0.2, 0.0, 0.0, 0.3, 0.4, 0.0, 0.0, 0.5, 0.6], dtype=np.float32),
    )


def test_write_wav_preserves_float_audio(monkeypatch, tmp_path) -> None:
    calls = []
    audio = np.array([0.1, -0.1], dtype=np.float32)
    path = tmp_path / "episode.wav"

    def fake_write(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr("podcast_llm.export.sf.write", fake_write)

    write_wav(path, audio, 44_100)

    assert calls == [((path, audio, 44_100), {"subtype": "FLOAT"})]


def test_maybe_export_mp3_uses_highest_quality_lame_vbr(monkeypatch, tmp_path) -> None:
    commands = []
    wav_path = tmp_path / "episode.wav"
    mp3_path = tmp_path / "episode.mp3"
    request = GenerationRequest(
        source_paths=[tmp_path / "notes.md"],
        language="en",
        duration_minutes=1,
        export_format="mp3",
    )

    def fake_run(command, **kwargs) -> None:
        commands.append((command, kwargs))

    monkeypatch.setattr("podcast_llm.export.shutil.which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr("podcast_llm.export.subprocess.run", fake_run)

    result = maybe_export_mp3(wav_path, mp3_path, request)

    assert result == mp3_path
    assert commands[0][0] == [
        "ffmpeg.exe",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "0",
        str(mp3_path),
    ]
