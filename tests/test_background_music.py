import numpy as np

from podcast_llm.background_music import (
    BASE_BACKGROUND_MUSIC_PROMPT,
    SectionMusicSpec,
    apply_edge_fades,
    build_section_prompt,
    fit_audio_to_length,
    mix_background_music,
)
from podcast_llm.models import DialogueTurn, OutlineSegment


def test_build_section_prompt_keeps_base_constraints_and_adds_section_context() -> None:
    segment = OutlineSegment(
        title="Hard Tradeoffs",
        summary="The hosts compare practical risks and decisions.",
        target_minutes=2,
    )
    turns = [
        DialogueTurn(speaker="Host A", text="This decision has real risk but a practical path."),
        DialogueTurn(speaker="Host B", text="The tradeoff is serious, but the outcome can be useful."),
    ]

    prompt = build_section_prompt(segment, turns)

    assert BASE_BACKGROUND_MUSIC_PROMPT in prompt
    assert "no vocals" in prompt
    assert "subtle and non-distracting" in prompt
    assert "Hard Tradeoffs" in prompt
    assert "analytical" in prompt or "light tension" in prompt


def test_fit_audio_to_length_loops_and_trims_to_exact_length() -> None:
    music = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    fitted = fit_audio_to_length(music, 8)

    np.testing.assert_array_equal(
        fitted,
        np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2], dtype=np.float32),
    )


def test_apply_edge_fades_preserves_length_and_fades_edges() -> None:
    audio = np.ones((10,), dtype=np.float32)

    faded = apply_edge_fades(audio, sample_rate=10, fade_seconds=0.2)

    assert len(faded) == 10
    assert faded[0] == 0
    assert faded[-1] == 0
    assert faded[3] == 1


def test_mix_background_music_uses_gain_and_prevents_clipping() -> None:
    speech = np.array([0.95, -0.95], dtype=np.float32)
    music = np.array([1.0, -1.0], dtype=np.float32)

    mixed = mix_background_music(speech, music, gain=0.2)

    assert mixed.dtype == np.float32
    assert float(np.max(np.abs(mixed))) <= 1.0
    assert mixed[0] > speech[0]


def test_section_music_spec_serializes_prompt_metadata() -> None:
    spec = SectionMusicSpec(index=1, title="Intro", prompt="quiet prompt", duration_seconds=12.5)

    assert spec.to_metadata() == {
        "index": 1,
        "title": "Intro",
        "prompt": "quiet prompt",
        "duration_seconds": 12.5,
    }
