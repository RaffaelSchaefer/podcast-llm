# Podcast LLM

Local podcast generator inspired by NotebookLM's podcast workflow. It parses source files to Markdown, asks a local [LM Studio](https://lmstudio.ai/) model (via the `lmstudio` SDK) for a segmented two-host script, and synthesizes the dialogue locally with Qwen3-TTS.

Audio generation uses `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` by default. Optional quiet dynamic background music uses [`ACE-Step/Ace-Step1.5`](https://huggingface.co/ACE-Step/Ace-Step1.5), which is published with an MIT license. WAV files preserve 32-bit float audio, and MP3 export uses ffmpeg/LAME's highest VBR quality setting.

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

Qwen TTS runs locally through PyTorch. Use `tts_mode: "cuda"` on NVIDIA machines, `tts_mode: "apple"` on Apple Silicon with PyTorch MPS, or leave it as `"auto"` to prefer CUDA and then Apple MPS. The first run may download Qwen model weights into the Hugging Face cache. If quiet dynamic background music is enabled, the first music run also downloads ACE-Step 1.5 weights into the Hugging Face cache.

## TUI

```powershell
uv run podcast-llm
```

The TUI adopts your terminal's own colour palette and lays out a step-by-step wizard with a persistent step rail down the left side. The rail shows your progress; click any step you've already visited to jump straight back to it.

1. **Sources** — a two-pane file browser: navigate to any folder (type a path in the location bar or use `..` / Backspace), press Enter on a file to add or remove it, and watch your picks build up in the "Chosen" pane. Grab many at once with "Add all in this folder", or paste a file, folder, or glob (e.g. `~/notes/*.md`) into the add field — the field also reaches files the browser hides (e.g. inside `.git` or `outputs`).
2. **Style & Format** — pick a style preset (Deep Dive, Quick Brief, Debate, Interview) to prefill the duration and tone, then adjust language, duration, and custom instructions.
3. **Voices & Engine** — choose the two Qwen speaker voices, TTS hardware mode, export format, optional quiet dynamic background music, LM Studio host, and model. The optional LM Studio host field overrides `PODCAST_LLM_HOST` (leave it blank to use the env / default). The model selector is prefilled from `PODCAST_LLM_MODEL` when set, tries to list downloaded models automatically, and can be refreshed with **Refresh models**. Leave the selector on the fallback option to use LM Studio's currently loaded / env-default model.
4. **Review** — confirm a summary, then watch live progress (parse → outline → script → synthesize → export) on a phase checklist with a progress bar and streaming log. When it finishes you can **Open folder** to reveal the output, **Restart** for a fresh episode, or **Quit**.

Use `Ctrl+N` / `Ctrl+B` (or the on-screen buttons) to move between steps and `Ctrl+Q` to quit.

## Non-Interactive Generation

Create a JSON config:

```json
{
  "source_paths": ["notes.md"],
  "language": "en",
  "duration_minutes": 8,
  "custom_instructions": "Keep the tone practical and include host disagreement before conclusions.",
  "host_a_voice": "Aiden",
  "host_b_voice": "Serena",
  "tts_mode": "auto",
  "qwen_tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "lmstudio_host": null,
  "export_format": "wav",
  "enable_background_music": false
}
```

Run:

```powershell
uv run podcast-llm generate --config config.json
```

Outputs are written to `outputs/<episode-slug>/` with the final audio at the top level and metadata under `metadata/`. WAV runs keep `episode.wav`; MP3 runs create `episode.mp3` when `ffmpeg` is installed and remove the intermediate WAV. `metadata/` contains `sources.md`, `outline.json`, `transcript.md`, and `run.json`. When `enable_background_music` is true, the pipeline builds an ACE-Step prompt for each script section from that section's title, summary, and final host dialogue while preserving the quiet no-vocals base prompt; those section prompts and mix settings are saved to `metadata/background_music.json`.
