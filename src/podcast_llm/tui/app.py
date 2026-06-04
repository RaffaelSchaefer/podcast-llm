from textual.app import App
from textual.binding import Binding

from .draft import PodcastDraft
from .screens.generation import GenerationScreen
from .screens.review import ReviewScreen
from .screens.sources import SourcesScreen
from .screens.style import StyleScreen
from .screens.voices import VoicesScreen


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
        self._max_index = 0

    def on_mount(self) -> None:
        # Adopt the terminal's own ANSI palette instead of a bespoke theme so the
        # UI matches whatever colours the user's terminal is configured with
        # (ansi-dark maps every theme variable onto the 16 ANSI colours).
        self.theme = "ansi-dark"
        self.push_screen(self.STEP_SCREENS[0]())

    def advance(self) -> None:
        if self._step_index < len(self.STEP_SCREENS) - 1:
            self._step_index += 1
            self._max_index = max(self._max_index, self._step_index)
            self.switch_screen(self.STEP_SCREENS[self._step_index]())

    def go_back(self) -> None:
        if self._step_index > 0:
            self._step_index -= 1
            self.switch_screen(self.STEP_SCREENS[self._step_index]())

    def goto_step(self, index: int) -> None:
        """Jump straight to a visited step from the rail.

        Only steps already reached (``index <= self._max_index``) are allowed;
        forward jumps stay gated behind per-step validation in ``action_next``.
        """
        if index == self._step_index or not (0 <= index <= self._max_index):
            return
        if isinstance(self.screen, self.STEP_SCREENS):
            try:
                self.screen.commit(strict=False)
            except ValueError:
                pass
        self._step_index = index
        self.switch_screen(self.STEP_SCREENS[index]())

    def restart(self) -> None:
        """Reset to a blank wizard for a fresh episode."""
        self.draft = PodcastDraft()
        self._step_index = 0
        self._max_index = 0
        self.switch_screen(self.STEP_SCREENS[0]())

    def start_generation(self) -> None:
        self.push_screen(GenerationScreen(self.draft.to_request()))
