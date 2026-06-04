# Podcast LLM

Local podcast generator inspired by NotebookLM's podcast workflow. It parses source files to Markdown, asks a local [LM Studio](https://lmstudio.ai/) model (via the `lmstudio` SDK) for a segmented two-host script, and synthesizes the dialogue locally with Supertonic.

Audio generation favors quality over speed: Supertonic runs at its maximum synthesis step count, WAV files preserve 32-bit float audio, and MP3 export uses ffmpeg/LAME's highest VBR quality setting.

## Setup

```powershell
uv sync
```

Create a `.env` file or set environment variables:

```powershell
$env:PODCAST_LLM_MODEL="qwen/qwen3.6-35b-a3b"
$env:PODCAST_LLM_HOST="100.124.83.79:1234"
```

Start the LM Studio server and load a model. `PODCAST_LLM_HOST` is the server's bare `host:port` (no scheme, no `/v1`); leave it unset to use LM Studio's default `localhost:1234`. `PODCAST_LLM_MODEL` is the model key to load — leave it unset to reuse whichever model is already loaded.

## TUI

```powershell
uv run podcast-llm
```

A step-by-step wizard walks you through four screens:

1. **Sources** — a two-pane file browser: navigate to any folder (type a path in the location bar or use `..` / Backspace), press Enter on a file to check it on or off, and watch your picks build up in the "Chosen" pane. Grab many at once with "Add all in this folder" or by pasting a file, folder, or glob (e.g. `~/notes/*.md`).
2. **Style & Format** — pick a style preset (Deep Dive, Quick Brief, Debate, Interview) to prefill the duration and tone, then adjust language, duration, and custom instructions.
3. **Voices & Engine** — choose the two host voices and export format. The model comes from `.env` / environment variables, while the optional LM Studio host field overrides `PODCAST_LLM_HOST` (leave it blank to use the env / default). Hit **Test connection & list models** to confirm the server is reachable.
4. **Review** — confirm a summary, then watch live progress (parse → outline → script → synthesize → export) with a progress bar and streaming log.

Use `Ctrl+N` / `Ctrl+B` (or the on-screen buttons) to move between steps and `Ctrl+Q` to quit.

## Non-Interactive Generation

Create a JSON config:

```json
{
  "source_paths": ["notes.md"],
  "language": "en",
  "duration_minutes": 8,
  "custom_instructions": "Keep the tone practical and include host disagreement before conclusions.",
  "host_a_voice": "M1",
  "host_b_voice": "F1",
  "lmstudio_host": null,
  "export_format": "wav"
}
```

Run:

```powershell
uv run podcast-llm generate --config config.json
```

Outputs are written to `outputs/<episode-slug>/` with the final audio at the top level and metadata under `metadata/`. WAV runs keep `episode.wav`; MP3 runs create `episode.mp3` when `ffmpeg` is installed and remove the intermediate WAV. `metadata/` contains `sources.md`, `outline.json`, `transcript.md`, and `run.json`.
