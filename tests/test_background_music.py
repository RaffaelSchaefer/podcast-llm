import numpy as np

import podcast_llm.background_music as background_music

from podcast_llm.background_music import (
    ACE_STEP_MODEL_ID,
    BASE_BACKGROUND_MUSIC_PROMPT,
    AceStepMusicGenerator,
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


def test_mix_background_music_normalizes_low_level_music_before_gain() -> None:
    speech = np.zeros((4,), dtype=np.float32)
    music = np.array([0.0001, -0.0001], dtype=np.float32)

    mixed = mix_background_music(speech, music, gain=0.2)

    assert np.isclose(float(np.max(np.abs(mixed))), 0.2)


def test_mix_background_music_leaves_silent_music_silent() -> None:
    speech = np.array([0.1, -0.1], dtype=np.float32)
    music = np.zeros((2,), dtype=np.float32)

    mixed = mix_background_music(speech, music, gain=0.2)

    np.testing.assert_array_equal(mixed, speech)


def test_section_music_spec_serializes_prompt_metadata() -> None:
    spec = SectionMusicSpec(index=1, title="Intro", prompt="quiet prompt", duration_seconds=12.5)

    assert spec.to_metadata() == {
        "index": 1,
        "title": "Intro",
        "prompt": "quiet prompt",
        "duration_seconds": 12.5,
    }


def test_ace_step_generator_uses_diffusers_pipeline_for_custom_ace_step_config(monkeypatch) -> None:
    calls = []

    class FakeAceStepPipeline:
        sample_rate = 4

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {"audios": np.array([[[0.0, 0.25, 0.5, 0.75]]], dtype=np.float32)}

    def broken_transformers_pipeline(*args, **kwargs):
        raise ValueError("Unrecognized configuration class AceStepConfig")

    def fake_load_ace_step_pipeline(model_id: str):
        calls.append({"loaded_model_id": model_id})
        return FakeAceStepPipeline()

    monkeypatch.setattr(background_music, "_transformers_pipeline", broken_transformers_pipeline)
    monkeypatch.setattr(background_music, "_load_ace_step_pipeline", fake_load_ace_step_pipeline, raising=False)

    audio = AceStepMusicGenerator().generate("quiet section music", duration_seconds=0.5, sample_rate=4)

    assert calls == [
        {"loaded_model_id": ACE_STEP_MODEL_ID},
        {
            "prompt": "quiet section music",
            "lyrics": "",
            "audio_duration": 10.0,
            "output_type": "np",
        },
    ]
    np.testing.assert_array_equal(audio, np.array([0.0, 0.25], dtype=np.float32))


def test_ace_step_generator_falls_back_when_diffusers_pipeline_metadata_is_missing(monkeypatch) -> None:
    def missing_pipeline_metadata(model_id: str):
        raise OSError(
            "Entry Not Found for url: "
            "https://huggingface.co/ACE-Step/acestep-v15-base/resolve/main/model_index.json"
        )

    monkeypatch.setattr(background_music, "_load_ace_step_pipeline", missing_pipeline_metadata, raising=False)

    audio = AceStepMusicGenerator().generate("quiet ambient piano", duration_seconds=0.25, sample_rate=8)

    assert audio.dtype == np.float32
    assert len(audio) == 2
    assert np.any(audio != 0)
