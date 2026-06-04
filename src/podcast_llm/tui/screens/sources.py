from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Input, Label, OptionList
from textual.widgets.option_list import Option

from ._paths import (
    IGNORED_SOURCE_DIRECTORIES,
    contains_path,
    expand_entry,
    is_hidden,
    path_key,
    split_tokens,
    supported_files_in,
)
from .base import WizardScreen


@dataclass
class _BrowseEntry:
    path: Path
    is_dir: bool


class SourcesScreen(WizardScreen):
    STEP_INDEX = 0
    CARD_TITLE = "Sources"

    BINDINGS = [Binding("backspace", "up_dir", "Up a folder", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self._current_dir = Path.cwd()
        self._browse_entries: list[_BrowseEntry] = []
        self._chosen_paths: list[Path] = []

    def compose_body(self) -> ComposeResult:
        yield Label(
            "Browse to a folder, then press Enter on a file to add or remove it. "
            "Enter on a folder opens it; Backspace goes up.",
            classes="hint",
        )
        yield Input(id="location", placeholder="Type or paste a folder, then Enter to jump there")
        with Horizontal(id="picker"):
            yield OptionList(id="browser")
            yield OptionList(id="chosen")
        with Horizontal(id="sources_actions"):
            yield Button("Add all in this folder", id="add_all")
            yield Button("Clear chosen", id="clear_chosen", variant="warning")
        yield Label("Add a file, folder, or glob (separate several with ;):", classes="section-label")
        yield Input(placeholder="e.g.  ~/notes/*.md  ;  C:\\refs\\paper.pdf", id="add_path")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#browser", OptionList).border_title = "Browse"
        self._load_dir(self._current_dir)
        self._refresh_chosen()

    def populate(self) -> None:
        self._refresh_chosen()

    def commit(self, strict: bool) -> None:
        if strict and not self.app.draft.source_paths:
            raise ValueError("Select at least one source file.")

    # --- navigation ---
    def navigate_to(self, directory: Path) -> None:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            self.set_status(f"Not a folder: {directory}")
            return
        self._load_dir(directory)

    def action_up_dir(self) -> None:
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._load_dir(parent)

    def _load_dir(self, directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            self.set_status(f"Cannot open {directory}: {exc}")
            return

        self._current_dir = directory
        self.query_one("#location", Input).value = str(directory)
        browser = self.query_one("#browser", OptionList)
        browser.clear_options()
        self._browse_entries = []
        options: list[Option] = []

        parent = directory.parent
        if parent != directory:
            self._browse_entries.append(_BrowseEntry(parent, is_dir=True))
            options.append(Option(Text("📁 ..", style="bold yellow")))

        for child in children:
            if child.is_dir():
                if child.name in IGNORED_SOURCE_DIRECTORIES or is_hidden(child):
                    continue
                self._browse_entries.append(_BrowseEntry(child, is_dir=True))
                label = Text("📁 ", style="bold yellow")
                label.append(f"{child.name}/")
                options.append(Option(label))
            elif child.is_file() and not is_hidden(child):
                self._browse_entries.append(_BrowseEntry(child, is_dir=False))
                options.append(Option(self._file_label(child)))

        browser.add_options(options)
        if options:
            browser.highlighted = 0

    def _file_label(self, path: Path) -> Text:
        selected = contains_path(self.app.draft.source_paths, path)
        label = Text()
        label.append("[x] " if selected else "[ ] ", style="bold green" if selected else "dim")
        label.append(path.name)
        return label

    # --- selection ---
    def add_source(self, path: Path) -> bool:
        if contains_path(self.app.draft.source_paths, path):
            return False
        self.app.draft.source_paths.append(Path(path))
        self._refresh_chosen()
        self._reload_markers()
        return True

    def toggle_source(self, path: Path) -> None:
        key = path_key(path)
        paths = self.app.draft.source_paths
        if contains_path(paths, path):
            self.app.draft.source_paths = [p for p in paths if path_key(p) != key]
            self.set_status(f"Removed {Path(path).name}")
        else:
            paths.append(Path(path))
            self.set_status(f"Added {Path(path).name}")
        self._refresh_chosen()
        self._reload_markers()

    def add_all_in_current_dir(self) -> int:
        added = sum(1 for path in supported_files_in(self._current_dir) if self.add_source(path))
        self.set_status(f"Added {added} file{'s' if added != 1 else ''} from this folder.")
        return added

    def add_by_pattern(self, text: str) -> int:
        added = 0
        matched = False
        for token in split_tokens(text):
            results = expand_entry(token, self._current_dir)
            matched = matched or bool(results)
            added += sum(1 for path in results if self.add_source(path))
        if not matched:
            self.set_status("No matching files found.")
        else:
            self.set_status(f"Added {added} file{'s' if added != 1 else ''}.")
        return added

    def clear_chosen(self) -> None:
        self.app.draft.source_paths = []
        self._refresh_chosen()
        self._reload_markers()
        self.set_status("Cleared all chosen sources.")

    # --- rendering ---
    def _refresh_chosen(self) -> None:
        chosen = self.query_one("#chosen", OptionList)
        chosen.clear_options()
        self._chosen_paths = list(self.app.draft.source_paths)
        options: list[Option] = []
        for path in self._chosen_paths:
            label = Text("✓ ", style="green")
            label.append(path.name)
            options.append(Option(label))
        chosen.add_options(options)
        chosen.border_title = f"Chosen ({len(self._chosen_paths)})"

    def _reload_markers(self) -> None:
        if not self.is_mounted:
            return
        browser = self.query_one("#browser", OptionList)
        highlighted = browser.highlighted
        self._load_dir(self._current_dir)
        if highlighted is not None and highlighted < browser.option_count:
            browser.highlighted = highlighted

    # --- events ---
    @on(OptionList.OptionSelected, "#browser")
    def _on_browse_selected(self, event: OptionList.OptionSelected) -> None:
        entry = self._browse_entries[event.option_index]
        if entry.is_dir:
            self.navigate_to(entry.path)
        else:
            self.toggle_source(entry.path)

    @on(OptionList.OptionSelected, "#chosen")
    def _on_chosen_selected(self, event: OptionList.OptionSelected) -> None:
        if 0 <= event.option_index < len(self._chosen_paths):
            self.toggle_source(self._chosen_paths[event.option_index])

    @on(Button.Pressed, "#add_all")
    def _on_add_all(self, event: Button.Pressed) -> None:
        self.add_all_in_current_dir()

    @on(Button.Pressed, "#clear_chosen")
    def _on_clear_chosen(self, event: Button.Pressed) -> None:
        self.clear_chosen()

    @on(Input.Submitted, "#location")
    def _on_location_submitted(self, event: Input.Submitted) -> None:
        self.navigate_to(Path(event.value))

    @on(Input.Submitted, "#add_path")
    def _on_add_path_submitted(self, event: Input.Submitted) -> None:
        self.add_by_pattern(event.value)
        event.input.value = ""
