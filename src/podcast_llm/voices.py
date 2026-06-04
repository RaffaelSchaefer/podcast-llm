"""Voice catalog helpers for the wizard's voice pickers.

Supertonic ships a fixed set of built-in voices (``M1``..``M5`` / ``F1``..``F5``).
The real installed list can only be read once the model is downloaded, so the
wizard defaults to this static catalog and offers an optional refresh.
"""

BUILTIN_VOICE_NAMES: tuple[str, ...] = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
)


def voice_label(name: str) -> str:
    """Friendly label for a voice name, e.g. ``M1`` -> ``M1 — Male voice 1``."""
    if len(name) >= 2 and name[0] in {"M", "F"} and name[1:].isdigit():
        gender = "Male" if name[0] == "M" else "Female"
        return f"{name} — {gender} voice {name[1:]}"
    return name


def voice_select_options(names: tuple[str, ...] | list[str] | None = None) -> list[tuple[str, str]]:
    """(label, value) pairs for a Textual ``Select`` of voices."""
    return [(voice_label(name), name) for name in (names or BUILTIN_VOICE_NAMES)]
