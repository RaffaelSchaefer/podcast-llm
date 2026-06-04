"""Voice catalog helpers for Qwen CustomVoice speakers."""

QWEN_VOICE_NAMES: tuple[str, ...] = (
    "Aiden",
    "Serena",
    "Ryan",
    "Vivian",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
)

QWEN_VOICE_DESCRIPTIONS = {
    "Aiden": "Sunny American male voice",
    "Serena": "Warm young female voice",
    "Ryan": "Dynamic English male voice",
    "Vivian": "Bright young female voice",
    "Uncle_Fu": "Seasoned low male voice",
    "Dylan": "Youthful Beijing male voice",
    "Eric": "Lively Chengdu male voice",
    "Ono_Anna": "Playful Japanese female voice",
    "Sohee": "Warm Korean female voice",
}


def voice_label(name: str) -> str:
    description = QWEN_VOICE_DESCRIPTIONS.get(name)
    if description:
        return f"{name} - {description}"
    return name


def voice_select_options(names: tuple[str, ...] | list[str] | None = None) -> list[tuple[str, str]]:
    return [(voice_label(name), name) for name in (names or QWEN_VOICE_NAMES)]
