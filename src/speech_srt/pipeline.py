"""
Pipeline Orchestrator Module.

Coordinates the end-to-end processing pipeline:
Validation -> Media Inspection -> Audio Extraction -> ASR -> Alignment -> Segmentation -> SRT Export.
Provides granular progress callbacks for GUI and CLI integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from speech_srt.config import AppConfig, load_config
from speech_srt.validators import validate_media_file


ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineResult:
    """Encapsulates output information returned from pipeline execution."""
    success: bool
    input_file: Path
    output_srt_file: Optional[Path] = None
    duration_seconds: float = 0.0
    subtitle_count: int = 0
    error_message: Optional[str] = None


class TranscriptionPipeline:
    """Orchestrates end-to-end subtitle generation from video input."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()

    def process(
        self,
        input_file: Path,
        output_srt_file: Optional[Path] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """
        Execute the full transcription pipeline on an input media file.

        Args:
            input_file: Path to source media file.
            output_srt_file: Optional custom destination Path for .srt file.
            progress_callback: Optional callable receiving (percentage: float, message: str).

        Returns:
            PipelineResult describing outcome.
        """
        input_path = Path(input_file)

        # 1. Validation
        is_valid, err_msg = validate_media_file(input_path)
        if not is_valid:
            return PipelineResult(
                success=False,
                input_file=input_path,
                error_message=f"Input validation failed: {err_msg}",
            )

        # Full pipeline stages will be orchestrated as sub-modules are completed in Milestones 2-4
        if progress_callback:
            progress_callback(10.0, "Input validated successfully.")

        return PipelineResult(
            success=False,
            input_file=input_path,
            error_message="Pipeline execution engine is scheduled for completion in Milestones 2-4.",
        )
