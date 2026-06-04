from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from .draft import PodcastDraft
from .screens.generation import GenerationScreen
from .screens.review import ReviewScreen
from .screens.sources import SourcesScreen
from .screens.style import StyleScreen
from .screens.voices import VoicesScreen

PODCAST_THEME = Theme(
    name="podcast",
    primary="#7C5CFF",
    secondary="#22D3EE",
    accent="#F59E0B",
    success="#34D399",
    warning="#FBBF24",
    error="#F87171",
    background="#0F1117",
    surface="#181B24",
    panel="#1F2430",
    dark=True,
    variables={
        "footer-key-foreground": "#7C5CFF",
        "block-cursor-background": "#7C5CFF",
        "input-selection-background": "#7C5CFF 35%",
    },
)


class PodcastWizard(App):
    CSS_PATH = "theme.tcss"
    TITLE = "Podcast LLM"
    SUB_TITLE = "Turn your sources into a two-host podcast"

    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]

    STEP_SCREENS = (SourcesScreen, StyleScreen, VoicesScreen, ReviewScreen)

    def __init__(self) -> None:
        super().__init__()
        self.draft = PodcastDraft()
        self._step_index = 0

    def on_mount(self) -> None:
        self.register_theme(PODCAST_THEME)
        self.theme = "podcast"
        self.push_screen(self.STEP_SCREENS[0]())

    def advance(self) -> None:
        if self._step_index < len(self.STEP_SCREENS) - 1:
            self._step_index += 1
            self.switch_screen(self.STEP_SCREENS[self._step_index]())

    def go_back(self) -> None:
        if self._step_index > 0:
            self._step_index -= 1
            self.switch_screen(self.STEP_SCREENS[self._step_index]())

    def start_generation(self) -> None:
        self.push_screen(GenerationScreen(self.draft.to_request()))
