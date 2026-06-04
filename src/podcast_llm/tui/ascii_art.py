"""Unicode-friendly ASCII animation frames for podcast generation.

The generation screen treats these as a stage: every phase owns a loop of
fixed-size frames, and the widget swaps loops as pipeline progress advances.
Frames are intentionally larger than the old compact panel so theater mode can
make the animation the primary UI.
"""

CANVAS_W = 72
CANVAS_H = 18

# Theme palette, kept close to PODCAST_THEME in app.py.
INK = "#22D3EE"
PEN = "#7C5CFF"
LENS = "#F59E0B"
ACCENT = "#FBBF24"
OK = "#34D399"
ERR = "#F87171"
DIM = "#5A6172"

Row = str | tuple[str, str]


def _frame(rows: list[Row]) -> str:
    """Pad rows to the shared canvas and apply optional Rich markup."""

    out: list[str] = []
    for row in rows:
        text, colour = row if isinstance(row, tuple) else (row, None)
        line = text[:CANVAS_W].ljust(CANVAS_W).replace("[", r"\[")
        out.append(f"[{colour}]{line}[/]" if colour else line)
    while len(out) < CANVAS_H:
        out.append(" " * CANVAS_W)
    return "\n".join(out[:CANVAS_H])


def _parse_frame(offset: int, beam: str) -> str:
    left = " " * offset
    return _frame(
        [
            ("        SOURCE SCANNER", INK),
            ("", None),
            ("   ╭────────────────────────────╮        ╭────────────╮", DIM),
            ("   │  page 01  page 02  page 03 │        │ index cards │", DIM),
            ("   │  ───────  ───────  ─────── │        │  ┌──┐ ┌──┐  │", DIM),
            ("   │  text..   text..   text..  │        │  └──┘ └──┘  │", DIM),
            ("   ╰────────────────────────────╯        ╰────────────╯", DIM),
            ("", None),
            (f"   {left}╭────────╮", LENS),
            (f"   {left}│  ◉  ◉  │ {beam}", LENS),
            (f"   {left}│   ▽    │ {beam}", LENS),
            (f"   {left}╰───┬────╯", LENS),
            (f"   {left}    ╲", LENS),
            (f"   {left}     ╲  scanning source text", ACCENT),
            ("", None),
            ("   extracting notes • normalizing markdown • sorting fragments", INK),
        ]
    )


_PARSE = [
    _parse_frame(0, "────"),
    _parse_frame(10, "──────"),
    _parse_frame(20, "────────"),
    _parse_frame(10, "──────"),
]


_OUTLINE = [
    _frame(
        [
            ("        OUTLINE DESK", INK),
            ("", None),
            ("   ╭─────────────────────────────────────────────╮", DIM),
            ("   │ Episode map                                 │", DIM),
            ("   │                                             │", DIM),
            ("   │  01  opening question  ───────────────╮     │", ACCENT),
            ("   │                                      ╱      │", ACCENT),
            ("   │                                ✎────╯      │", PEN),
            ("   │                                             │", DIM),
            ("   │  02                                         │", DIM),
            ("   │  03                                         │", DIM),
            ("   ╰─────────────────────────────────────────────╯", DIM),
            ("", None),
            ("   pen pressure: light • structure: forming • cards: 1/4", INK),
        ]
    ),
    _frame(
        [
            ("        OUTLINE DESK", INK),
            ("", None),
            ("   ╭─────────────────────────────────────────────╮", DIM),
            ("   │ Episode map                                 │", DIM),
            ("   │                                             │", DIM),
            ("   │  01  opening question                       │", INK),
            ("   │  02  core explanation  ───────────────╮     │", ACCENT),
            ("   │                                      ╱      │", ACCENT),
            ("   │                                ✎────╯      │", PEN),
            ("   │  03                                         │", DIM),
            ("   │  04                                         │", DIM),
            ("   ╰─────────────────────────────────────────────╯", DIM),
            ("", None),
            ("   pen pressure: steady • structure: linking • cards: 2/4", INK),
        ]
    ),
    _frame(
        [
            ("        OUTLINE DESK", INK),
            ("", None),
            ("   ╭─────────────────────────────────────────────╮", DIM),
            ("   │ Episode map                                 │", DIM),
            ("   │                                             │", DIM),
            ("   │  01  opening question                       │", INK),
            ("   │  02  core explanation                       │", INK),
            ("   │  03  host disagreement ───────────────╮     │", ACCENT),
            ("   │                                      ╱      │", ACCENT),
            ("   │                                ✎────╯      │", PEN),
            ("   │  04                                         │", DIM),
            ("   ╰─────────────────────────────────────────────╯", DIM),
            ("", None),
            ("   pen pressure: quick • structure: tension • cards: 3/4", INK),
        ]
    ),
    _frame(
        [
            ("        OUTLINE DESK", INK),
            ("", None),
            ("   ╭─────────────────────────────────────────────╮", DIM),
            ("   │ Episode map                                 │", DIM),
            ("   │                                             │", DIM),
            ("   │  01  opening question                       │", INK),
            ("   │  02  core explanation                       │", INK),
            ("   │  03  host disagreement                      │", INK),
            ("   │  04  final takeaways ────────────────╮      │", ACCENT),
            ("   │                                     ╱       │", ACCENT),
            ("   │                               ✎────╯       │", PEN),
            ("   ╰─────────────────────────────────────────────╯", DIM),
            ("", None),
            ("   pen pressure: done • structure: locked • cards: 4/4", OK),
        ]
    ),
]


_SCRIPT = [
    _frame(
        [
            ("        SCRIPT ROOM", INK),
            ("", None),
            ("              ╭──────────────────────────╮", DIM),
            ("              │ dialogue sheet           │", DIM),
            ("              │ A: Today we open with _  │", PEN),
            ("              │                          │", DIM),
            ("              │                          │", DIM),
            ("              ╰──────────────────────────╯", DIM),
            ("", None),
            ("        ╭────────────────────────────────────╮", ACCENT),
            ("        │  type hammers rise:  t  a  l  k   │", ACCENT),
            ("        ╰──────┬─────────────────────┬──────╯", ACCENT),
            ("               ●                     ●", DIM),
            ("", None),
            ("        carriage: left • page fill: first line", INK),
        ]
    ),
    _frame(
        [
            ("        SCRIPT ROOM", INK),
            ("", None),
            ("              ╭──────────────────────────╮", DIM),
            ("              │ dialogue sheet           │", DIM),
            ("              │ A: Today we open with    │", INK),
            ("              │ B: Then I push back _    │", PEN),
            ("              │                          │", DIM),
            ("              ╰──────────────────────────╯", DIM),
            ("", None),
            ("        ╭────────────────────────────────────╮", ACCENT),
            ("        │     clack  clack  clack  clack     │", ACCENT),
            ("        ╰────────────┬───────────────┬───────╯", ACCENT),
            ("                     ●               ●", DIM),
            ("", None),
            ("        carriage: center • page fill: exchange forming", INK),
        ]
    ),
    _frame(
        [
            ("        SCRIPT ROOM", INK),
            ("", None),
            ("              ╭──────────────────────────╮", DIM),
            ("              │ dialogue sheet           │", DIM),
            ("              │ A: Today we open with    │", INK),
            ("              │ B: Then I push back      │", INK),
            ("              │ A: Exactly, because _    │", PEN),
            ("              ╰──────────────────────────╯", DIM),
            ("", None),
            ("        ╭────────────────────────────────────╮", ACCENT),
            ("        │        line break • return         │", ACCENT),
            ("        ╰────────────────┬────────────┬──────╯", ACCENT),
            ("                         ●            ●", DIM),
            ("", None),
            ("        carriage: right • page fill: rhythm set", INK),
        ]
    ),
    _frame(
        [
            ("        SCRIPT ROOM", INK),
            ("", None),
            ("              ╭──────────────────────────╮", DIM),
            ("              │ dialogue sheet           │", DIM),
            ("              │ B: Then I push back      │", INK),
            ("              │ A: Exactly, because      │", INK),
            ("              │ B: So the listener gets _│", PEN),
            ("              ╰──────────────────────────╯", DIM),
            ("", None),
            ("        ╭────────────────────────────────────╮", ACCENT),
            ("        │   page advances, fresh line ready  │", ACCENT),
            ("        ╰──────┬─────────────────────┬──────╯", ACCENT),
            ("               ●                     ●", DIM),
            ("", None),
            ("        carriage: reset • page fill: next turn", INK),
        ]
    ),
]


def _synth_frame(active: str, waveform: str, bubble: str) -> str:
    if active == "A":
        rows = [
            ("        VOICE BOOTH", INK),
            ("", None),
            ("        ╭────────────────╮", ACCENT),
            (f"       ( {bubble:<14} )", ACCENT),
            ("        ╰──────┬─────────╯", ACCENT),
            ("               │", ACCENT),
            ("          ◉────┘                         ○", OK),
            ("        ╭────╮                         ╭────╮", INK),
            ("        │ A  │                         │ B  │", INK),
            ("        ╰────╯                         ╰────╯", INK),
            ("        HOST A                         HOST B", DIM),
            ("", None),
            (f"        {waveform}", LENS),
            ("        routing text to voice model • speaker: HOST A", OK),
        ]
    else:
        rows = [
            ("        VOICE BOOTH", INK),
            ("", None),
            ("                                      ╭────────────────╮", ACCENT),
            (f"                                     ( {bubble:<14} )", ACCENT),
            ("                                      ╰──────┬─────────╯", ACCENT),
            ("                                             │", ACCENT),
            ("          ○                                  └────◉", OK),
            ("        ╭────╮                         ╭────╮", INK),
            ("        │ A  │                         │ B  │", INK),
            ("        ╰────╯                         ╰────╯", INK),
            ("        HOST A                         HOST B", DIM),
            ("", None),
            (f"        {waveform}", LENS),
            ("        routing text to voice model • speaker: HOST B", OK),
        ]
    return _frame(rows)


_SYNTH_A = [
    _synth_frame("A", "▁▂▃▅▆▇▆▅▃▂▁  ▁▂▃▅▆▇▆▅▃▂▁", "opening line"),
    _synth_frame("A", "▂▃▅▇▇▅▃▂▁    ▂▃▅▇▇▅▃▂", "emphasis"),
    _synth_frame("A", "▁▁▂▃▅▆▅▃▂▁    ▁▁▂▃▅▆", "breath"),
]

_SYNTH_B = [
    _synth_frame("B", "▁▂▃▅▆▇▆▅▃▂▁  ▁▂▃▅▆▇▆▅▃▂▁", "reply"),
    _synth_frame("B", "▂▃▅▇▇▅▃▂▁    ▂▃▅▇▇▅▃▂", "counterpoint"),
    _synth_frame("B", "▁▁▂▃▅▆▅▃▂▁    ▁▁▂▃▅▆", "closing beat"),
]


_EXPORT = [
    _frame(
        [
            ("        EXPORT FACTORY", INK),
            ("", None),
            ("             smoke:  ·   ·", ACCENT),
            ("        ╭────────────────────────────╮", DIM),
            ("        │  AUDIO PRESSING FLOOR      │", PEN),
            ("        │  mix → master → package    │", PEN),
            ("        ╰──────────────┬─────────────╯", DIM),
            ("                       ╱", DIM),
            ("        ════════╦═════╩═════╦════════════════════", LENS),
            ("                ║  WAV BOX  ║", LENS),
            ("        ○   ○   ╚═══════════╝   ○   ○   ○", DIM),
            ("", None),
            ("        transcript stamped • waveform packed • format queued", INK),
        ]
    ),
    _frame(
        [
            ("        EXPORT FACTORY", INK),
            ("", None),
            ("             smoke:  ·   ○   ·", ACCENT),
            ("        ╭────────────────────────────╮", DIM),
            ("        │  AUDIO PRESSING FLOOR      │", PEN),
            ("        │  mix → master → package    │", PEN),
            ("        ╰──────────────┬─────────────╯", DIM),
            ("                       ╱", DIM),
            ("        ═══════════════╦═════╩═════╦═════════════", LENS),
            ("                       ║ AUDIO BOX ║", LENS),
            ("        ○   ○   ○      ╚═══════════╝   ○   ○", DIM),
            ("", None),
            ("        transcript stamped • waveform packed • moving out", INK),
        ]
    ),
    _frame(
        [
            ("        EXPORT FACTORY", INK),
            ("", None),
            ("             smoke:  ○   ·   ○", ACCENT),
            ("        ╭────────────────────────────╮", DIM),
            ("        │  AUDIO PRESSING FLOOR      │", PEN),
            ("        │  mix → master → package    │", PEN),
            ("        ╰──────────────┬─────────────╯", DIM),
            ("                       ╱", DIM),
            ("        ═══════════════════════╦═════╩═════╦═════", LENS),
            ("                               ║ MP3 / WAV ║", LENS),
            ("        ○   ○   ○   ○          ╚═══════════╝   ○", DIM),
            ("", None),
            ("        transcript stamped • waveform packed • final pass", INK),
        ]
    ),
]


_DONE = [
    _frame(
        [
            ("", None),
            ("        ╭────────────────────────────────────────────╮", OK),
            ("        │                                            │", OK),
            ("        │         EPISODE READY                      │", OK),
            ("        │                                            │", OK),
            ("        │        ╭──────╮    ╭──────╮               │", OK),
            ("        │        │  L   │────│  R   │               │", OK),
            ("        │        ╰──────╯    ╰──────╯               │", OK),
            ("        │             final audio saved              │", OK),
            ("        │                                            │", OK),
            ("        ╰────────────────────────────────────────────╯", OK),
            ("", None),
            ("        press L for the log, or return to review when needed", INK),
        ]
    ),
    _frame(
        [
            ("", None),
            ("        ╭────────────────────────────────────────────╮", OK),
            ("        │       ✦                                    │", ACCENT),
            ("        │         EPISODE READY                 ✦    │", OK),
            ("        │                                            │", OK),
            ("        │        ╭──────╮    ╭──────╮               │", OK),
            ("        │        │  L   │────│  R   │               │", OK),
            ("        │        ╰──────╯    ╰──────╯               │", OK),
            ("        │             final audio saved              │", OK),
            ("        │                                    ✦       │", ACCENT),
            ("        ╰────────────────────────────────────────────╯", OK),
            ("", None),
            ("        press L for the log, or return to review when needed", INK),
        ]
    ),
]


_FAILED = [
    _frame(
        [
            ("", None),
            ("        ╭────────────────────────────────────────────╮", ERR),
            ("        │                                            │", ERR),
            ("        │             GENERATION STOPPED             │", ERR),
            ("        │                                            │", ERR),
            ("        │          ╭──────────────╮                  │", ERR),
            ("        │          │  error log   │ ◀ press L        │", ERR),
            ("        │          ╰──────────────╯                  │", ERR),
            ("        │                                            │", ERR),
            ("        ╰────────────────────────────────────────────╯", ERR),
            ("", None),
            ("        open log view for the exact failure details", DIM),
        ]
    )
]


FRAMES: dict[str, list[str]] = {
    "parse": _PARSE,
    "outline": _OUTLINE,
    "script": _SCRIPT,
    "synthesize": _SYNTH_A,
    "synthesize:A": _SYNTH_A,
    "synthesize:B": _SYNTH_B,
    "export": _EXPORT,
    "done": _DONE,
    "failed": _FAILED,
}
