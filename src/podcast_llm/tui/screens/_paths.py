"""Source-path helpers for the Sources step."""

from pathlib import Path

IGNORED_SOURCE_DIRECTORIES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "outputs",
}

# Document types MarkItDown handles well — used for bulk "add folder"/glob grabs.
# Manual picks in the browser are not restricted to these.
SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".text",
    ".rst",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".rtf",
    ".epub",
}

_GLOB_CHARS = ("*", "?", "[")


def path_key(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def contains_path(paths: list[Path], path: Path) -> bool:
    key = path_key(path)
    return any(path_key(existing) == key for existing in paths)


def is_hidden(path: Path) -> bool:
    return Path(path).name.startswith(".")


def human_size(path: Path) -> str:
    """Best-effort human-readable file size, or "" if it cannot be read."""
    try:
        size = float(Path(path).stat().st_size)
    except OSError:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def is_supported(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def split_tokens(text: str) -> list[str]:
    return [part.strip() for part in text.split(";") if part.strip()]


def supported_files_in(directory: Path) -> list[Path]:
    try:
        children = sorted(Path(directory).iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    return [p for p in children if p.is_file() and is_supported(p) and not is_hidden(p)]


def expand_entry(token: str, base_dir: Path) -> list[Path]:
    """Resolve a user-typed token into concrete files.

    Handles a glob pattern, a directory (all supported files within), or a
    single file. Paths are resolved relative to ``base_dir`` when not absolute.
    Returns an empty list when nothing matches.
    """
    token = token.strip()
    if not token:
        return []
    expanded = Path(token).expanduser()
    if any(char in token for char in _GLOB_CHARS):
        try:
            if expanded.is_absolute():
                anchor = Path(expanded.anchor)
                matches = anchor.glob(str(expanded.relative_to(anchor)))
            else:
                matches = Path(base_dir).glob(token)
            return sorted(match for match in matches if match.is_file())
        except (OSError, ValueError):
            return []
    path = expanded if expanded.is_absolute() else (Path(base_dir) / expanded)
    if path.is_dir():
        return supported_files_in(path)
    if path.is_file():
        return [path]
    return []
