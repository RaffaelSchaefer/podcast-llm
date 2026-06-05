from pathlib import Path

from .export import concat_audio, maybe_export_mp3, slugify, write_json, write_transcript, write_wav
from .llm import LLMProvider
from .models import DialogueTurn, EpisodeOutline, GenerationRequest, GenerationResult
from .parser import MarkItDownParser
from .progress import ProgressCallback, ProgressEvent
from .tts import QwenSynthesizer


class PodcastPipeline:
    def __init__(
        self,
        parser: MarkItDownParser | None = None,
        llm_provider: LLMProvider | None = None,
        synthesizer: QwenSynthesizer | None = None,
    ) -> None:
        self.parser = parser or MarkItDownParser()
        self.llm_provider = llm_provider
        self.synthesizer = synthesizer

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        def emit(event: ProgressEvent) -> None:
            if progress is not None:
                progress(event)

        if self.llm_provider is None:
            from .llm import provider_for_request

            self.llm_provider = provider_for_request(request)
        if self.synthesizer is None:
            self.synthesizer = QwenSynthesizer(request)

        emit(ProgressEvent("parse", "Parsing source files…"))
        parsed_sources = self.parser.parse(request.source_paths, request=request)

        emit(ProgressEvent("outline", "Designing the episode outline…"))
        emit(ProgressEvent("outline", kind="text_reset", text_scope="outline"))
        outline = self.llm_provider.make_outline(
            parsed_sources,
            request,
            text_progress=lambda text: emit(
                ProgressEvent("outline", kind="text_fragment", text=text, text_scope="outline")
            ),
        )
        emit(
            ProgressEvent(
                "outline",
                kind="text_replace",
                text=_format_outline(outline),
                text_scope="outline",
            )
        )

        output_dir = self._unique_output_dir(request.output_dir / slugify(outline.title))
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        (metadata_dir / "sources.md").write_text(parsed_sources.markdown, encoding="utf-8")
        write_json(metadata_dir / "outline.json", outline.model_dump(mode="json"))

        segment_count = len(outline.segments)
        emit(ProgressEvent("script", f"Outline ready — {segment_count} segments to write.", 0, segment_count))

        turns_by_segment: list[list[DialogueTurn]] = []
        prior_context = ""
        total_turns = 0

        for segment_index, segment in enumerate(outline.segments, start=1):
            emit(
                ProgressEvent(
                    "script",
                    kind="text_reset",
                    text_scope="script",
                    seg=segment_index,
                    segment_title=segment.title,
                )
            )
            turns = self.llm_provider.make_script_segment(
                segment,
                prior_context,
                request,
                text_progress=lambda text, segment_index=segment_index, segment=segment: emit(
                    ProgressEvent(
                        "script",
                        kind="text_fragment",
                        text=text,
                        text_scope="script",
                        seg=segment_index,
                        segment_title=segment.title,
                    )
                ),
            )
            turns_by_segment.append(turns)
            prior_context = " ".join(turn.text for turn in turns[-4:])
            total_turns += len(turns)
            emit(
                ProgressEvent(
                    "script",
                    kind="text_replace",
                    text=_format_turns(turns),
                    text_scope="script",
                    seg=segment_index,
                    segment_title=segment.title,
                )
            )
            emit(ProgressEvent("script", f"Scripted segment {segment_index}/{segment_count}: {segment.title}", segment_index, segment_count))

        audio_chunks = []
        turn_index = 0
        for segment_index, turns in enumerate(turns_by_segment, start=1):
            for turn_in_seg, turn in enumerate(turns, start=1):
                turn_index += 1
                emit(
                    ProgressEvent(
                        "synthesize",
                        f"Turn {turn_index}/{total_turns}",
                        turn_index,
                        total_turns,
                        seg=segment_index,
                        round_current=turn_in_seg,
                        round_total=len(turns),
                        speaker=turn.speaker,
                        delivery_instruction=turn.delivery_instruction,
                        text=turn.text,
                    )
                )
                chunk = self.synthesizer.synthesize_turn(turn, request)
                audio_chunks.append(chunk)

        emit(ProgressEvent("export", "Writing transcript and assembling the episode…"))
        transcript_path = metadata_dir / "transcript.md"
        write_transcript(transcript_path, outline, turns_by_segment)

        wav_path = output_dir / "episode.wav"
        write_wav(wav_path, concat_audio(audio_chunks, self.synthesizer.sample_rate), self.synthesizer.sample_rate)
        mp3_path = maybe_export_mp3(wav_path, output_dir / "episode.mp3", request)
        audio_path = mp3_path or wav_path
        if mp3_path is not None and wav_path.exists():
            wav_path.unlink()

        write_json(
            metadata_dir / "run.json",
            {
                "request": request.model_dump(mode="json"),
                "outputs": {
                    "transcript": str(transcript_path),
                    "audio": str(audio_path),
                    "mp3": str(mp3_path) if mp3_path else None,
                },
            },
        )
        emit(ProgressEvent("done", f"Saved episode to {output_dir}", total_turns, total_turns))
        return GenerationResult(
            output_dir=output_dir,
            transcript_path=transcript_path,
            audio_path=audio_path,
            mp3_path=mp3_path,
        )

    @staticmethod
    def _unique_output_dir(base: Path) -> Path:
        if not base.exists():
            return base
        suffix = 2
        while True:
            candidate = Path(f"{base}-{suffix}")
            if not candidate.exists():
                return candidate
            suffix += 1


def _format_outline(outline: EpisodeOutline) -> str:
    lines = [outline.title]
    for index, segment in enumerate(outline.segments, start=1):
        lines.append(f"{index}. {segment.title} - {segment.summary}")
    return "\n".join(lines)


def _format_turns(turns: list[DialogueTurn]) -> str:
    return "\n\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)
