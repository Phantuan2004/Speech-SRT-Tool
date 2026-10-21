"""
Pipeline Orchestrator Module.

Coordinates the end-to-end subtitle generation process:
1. Input validation
2. FFprobe media inspection & audio stream verification
3. FFmpeg audio extraction (16kHz mono WAV)
4. Offline ASR transcription with word-level timestamps (faster-whisper)
5. Sentence segmentation & CapCut layout wrapping
6. Validation checks (chronology, duration, overlaps)
7. Atomic UTF-8 SRT writing and verification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional
from speech_srt.asr_engine import FasterWhisperEngine
from speech_srt.audio_extractor import extract_audio_temp
from speech_srt.config import AppConfig, load_config
from speech_srt.media_info import MediaInspectionError, NoAudioStreamError, get_media_info
from speech_srt.sentence_segmenter import SubtitleBlock, segment_transcription_result
from speech_srt.srt_writer import parse_srt, write_srt
from speech_srt.validators import (
    ValidationReport,
    validate_media_file,
    validate_subtitle_blocks,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineResult:
    """Encapsulates output information returned from pipeline execution."""
    success: bool
    input_file: Path
    output_srt_file: Optional[Path] = None
    duration_seconds: float = 0.0
    subtitle_count: int = 0
    subtitles: List[SubtitleBlock] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class TranscriptionPipeline:
    """Orchestrates end-to-end subtitle generation without GUI dependencies."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.model_size = model_size or self.config.asr.model_size
        self.device = device or self.config.asr.device
        self.compute_type = compute_type or self.config.asr.compute_type
        self.offline = offline

        # Lazy engine container
        self._engine: Optional[FasterWhisperEngine] = None

    def _get_engine(self) -> FasterWhisperEngine:
        """Lazily initialize the ASR engine instance."""
        if self._engine is None:
            self._engine = FasterWhisperEngine(
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._engine

    def process(
        self,
        input_file: Path,
        output_srt_file: Optional[Path] = None,
        progress_callback: Optional[ProgressCallback] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> PipelineResult:
        """
        Execute the complete subtitle generation pipeline on an input media file.

        Args:
            input_file: Source video or audio path.
            output_srt_file: Custom destination path for the exported .srt file.
            progress_callback: Optional callback receiving (percentage: float, message: str).
            beam_size: Beam search width for transcription.
            vad_filter: Whether to apply Silero VAD filtering.

        Returns:
            PipelineResult instance with status, paths, and metadata.
        """
        input_path = Path(input_file).resolve()

        def update_progress(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)
            logger.info("[%3.0f%%] %s", pct, msg)

        # 1. Validate input file
        update_progress(5.0, f"Validating input file: '{input_path.name}'...")
        is_valid, err_msg = validate_media_file(input_path)
        if not is_valid:
            return PipelineResult(
                success=False,
                input_file=input_path,
                error_message=f"Input validation error: {err_msg}",
            )

        # 2. Determine destination SRT file
        if output_srt_file is not None:
            target_srt = Path(output_srt_file).resolve()
        else:
            out_dir = self.config.resolve_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            target_srt = out_dir / f"{input_path.stem}.srt"

        # 3. Inspect media info and audio stream
        update_progress(12.0, "Inspecting media streams via FFprobe...")
        try:
            media_info = get_media_info(input_path, require_audio=True)
        except (NoAudioStreamError, MediaInspectionError, FileNotFoundError) as exc:
            return PipelineResult(
                success=False,
                input_file=input_path,
                error_message=str(exc),
            )

        update_progress(
            20.0,
            f"Media verified (Duration: {media_info.duration_seconds:.1f}s, Audio: {media_info.audio_codec})."
        )

        # 4. Extract audio track to temporary WAV
        update_progress(25.0, "Extracting 16kHz mono audio via FFmpeg...")
        try:
            with extract_audio_temp(
                input_video=input_path,
                sample_rate=self.config.audio_extraction.sample_rate,
                channels=self.config.audio_extraction.channels,
            ) as wav_path:
                update_progress(35.0, "Audio extracted. Preparing ASR engine...")

                # 5. Load model and transcribe
                engine = self._get_engine()
                update_progress(
                    45.0,
                    f"Loading ASR model '{self.model_size}' on {engine.device} ({engine.compute_type})..."
                )
                engine.load_model(offline=self.offline)

                update_progress(55.0, "Transcribing speech and extracting word-level timestamps...")
                asr_result = engine.transcribe(
                    audio_path=wav_path,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=True,
                )

        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=input_path,
                duration_seconds=media_info.duration_seconds,
                error_message=f"Processing failed during extraction or transcription: {exc}",
            )

        # 6. Sentence Segmentation & CapCut layout wrapping
        update_progress(80.0, "Segmenting transcription into natural sentence cues...")
        cues = segment_transcription_result(
            result=asr_result.segments,
            max_line_length=self.config.subtitles.max_line_length,
            max_lines_per_subtitle=self.config.subtitles.max_lines_per_subtitle,
        )

        # 7. Validate generated subtitle blocks
        update_progress(90.0, "Validating subtitle timestamps and continuity...")
        val_report: ValidationReport = validate_subtitle_blocks(
            blocks=cues,
            video_duration=media_info.duration_seconds,
        )

        if not val_report.is_valid:
            error_details = "; ".join(val_report.errors)
            return PipelineResult(
                success=False,
                input_file=input_path,
                duration_seconds=media_info.duration_seconds,
                subtitle_count=len(cues),
                subtitles=cues,
                warnings=val_report.warnings,
                error_message=f"Subtitle validation failed: {error_details}",
            )

        # 8. Export atomic SRT
        update_progress(95.0, f"Writing CapCut-compliant SRT to '{target_srt.name}'...")
        try:
            write_srt(cues, target_srt)
            # Re-parse to guarantee round-trip integrity
            parse_srt(target_srt)
        except Exception as exc:
            return PipelineResult(
                success=False,
                input_file=input_path,
                duration_seconds=media_info.duration_seconds,
                error_message=f"Failed writing or verifying SRT file: {exc}",
            )

        update_progress(100.0, f"Completed successfully! {len(cues)} subtitle cues exported.")

        return PipelineResult(
            success=True,
            input_file=input_path,
            output_srt_file=target_srt,
            duration_seconds=media_info.duration_seconds,
            subtitle_count=len(cues),
            subtitles=cues,
            warnings=val_report.warnings,
        )
