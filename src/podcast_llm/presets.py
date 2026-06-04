"""Podcast style presets that prefill duration and tone in the wizard.

Selecting a preset seeds the duration and custom-instruction fields; the user
can still edit them freely afterward. ``CUSTOM_PRESET_ID`` is the default and
seeds nothing.
"""

from dataclasses import dataclass

CUSTOM_PRESET_ID = "custom"


@dataclass(frozen=True)
class PodcastStylePreset:
    id: str
    label: str
    description: str
    duration_minutes: int
    custom_instructions: str


PRESETS: tuple[PodcastStylePreset, ...] = (
    PodcastStylePreset(
        id="deep_dive",
        label="Deep Dive",
        description="Thorough, analytical exploration of the material.",
        duration_minutes=15,
        custom_instructions=(
            "Take a thorough, analytical approach. Define key terms, explore nuances and "
            "trade-offs, and connect ideas across the sources. Let the hosts ask follow-up "
            "questions and build on each other before reaching conclusions."
        ),
    ),
    PodcastStylePreset(
        id="quick_brief",
        label="Quick Brief",
        description="Punchy, high-signal summary of the essentials.",
        duration_minutes=5,
        custom_instructions=(
            "Keep it short, punchy, and high-signal. Lead with the most important takeaways, "
            "skip tangents, and keep each turn brief and concrete."
        ),
    ),
    PodcastStylePreset(
        id="debate",
        label="Debate",
        description="Hosts argue opposing sides before finding common ground.",
        duration_minutes=12,
        custom_instructions=(
            "Frame the episode as a friendly debate. Have Host A and Host B take opposing "
            "positions, challenge each other's reasoning with evidence from the sources, and "
            "only reach a nuanced synthesis near the end."
        ),
    ),
    PodcastStylePreset(
        id="interview",
        label="Interview",
        description="One host interviews the other as the subject expert.",
        duration_minutes=10,
        custom_instructions=(
            "Frame the episode as an interview. Host A asks focused, curious questions and "
            "Host B answers as the subject-matter expert, going deeper whenever Host A probes."
        ),
    ),
)


def preset_by_id(preset_id: str | None) -> PodcastStylePreset | None:
    for preset in PRESETS:
        if preset.id == preset_id:
            return preset
    return None
