import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .models import GenerationRequest, GenerationResult


def load_request_from_config(path: Path) -> GenerationRequest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GenerationRequest.model_validate(payload)


def run_generate(request: GenerationRequest) -> GenerationResult:
    from .pipeline import PodcastPipeline

    return PodcastPipeline().generate(request)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="podcast-llm")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Generate from a JSON config file")
    generate.add_argument("--config", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "generate":
        result = run_generate(load_request_from_config(args.config))
        print(f"Transcript: {result.transcript_path}")
        print(f"Audio: {result.audio_path}")
        return

    from .tui import PodcastWizard

    PodcastWizard().run()
