"""
Validation Utilities Module.

Validates input files, timestamp coherence, duration bounds, and SRT cues.
Returns structured error and warning reports without destructive auto-modifications.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
from speech_srt.sentence_segmenter import SubtitleBlock

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"
}
SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"
}
SUPPORTED_EXTENSIONS: Set[str] = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS


@dataclass
class ValidationReport:
    """Detailed summary of validation errors and warnings."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_media_file(path: Path) -> Tuple[bool, str]:
    """
    Validate that an input media file exists, is non-empty, and has a supported extension.

    Args:
        path: Path to the media file.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    file_path = Path(path)
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a regular file: {file_path}"

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported file extension '{suffix}'. Supported formats: {supported_list}"

    try:
        size = file_path.stat().st_size
        if size == 0:
            return False, f"File is empty (0 bytes): {file_path}"
    except OSError as err:
        return False, f"Cannot inspect file size: {err}"

    return True, "Valid media file"


def validate_timestamps(
    start_time: float,
    end_time: float,
    max_duration: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Validate that start and end timestamps form a valid, non-negative interval.

    Args:
        start_time: Interval start in seconds.
        end_time: Interval end in seconds.
        max_duration: Optional media duration bound.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if start_time < 0.0:
        return False, f"Start time cannot be negative: {start_time:.3f}s"

    if end_time < 0.0:
        return False, f"End time cannot be negative: {end_time:.3f}s"

    if end_time <= start_time:
        return False, f"End time ({end_time:.3f}s) must be strictly greater than start time ({start_time:.3f}s)"

    if max_duration is not None and end_time > max_duration + 0.5:
        return False, f"End time ({end_time:.3f}s) exceeds media duration ({max_duration:.3f}s)"

    return True, "Valid timestamp range"


def validate_subtitle_blocks(
    blocks: Sequence[SubtitleBlock],
    video_duration: Optional[float] = None,
) -> ValidationReport:
    """
    Perform deep validation of subtitle cues:
    - Empty cue text
    - Negative timestamps or start >= end
    - Chronological progression order
    - Abnormal overlaps between consecutive cues
    - Cues exceeding media duration

    Args:
        blocks: Sequence of SubtitleBlock instances.
        video_duration: Optional duration of media in seconds.

    Returns:
        ValidationReport instance.
    """
    report = ValidationReport(is_valid=True)

    if not blocks:
        if video_duration and video_duration > 1.0:
            report.add_warning(
                f"Generated subtitle list is empty, but media duration is {video_duration:.1f}s."
            )
        return report

    prev_end = 0.0
    prev_start = -1.0

    for idx, block in enumerate(blocks, start=1):
        # Index continuity
        if block.index != idx:
            report.add_error(f"Cue index mismatch at position {idx}: expected {idx}, got {block.index}")

        # Empty text check
        if not block.text.strip():
            report.add_error(f"Cue #{idx} contains empty text (start: {block.start_time:.3f}s)")

        # Timestamp bounds
        if block.start_time < 0.0:
            report.add_error(f"Cue #{idx} has negative start time: {block.start_time:.3f}s")

        if block.end_time < 0.0:
            report.add_error(f"Cue #{idx} has negative end time: {block.end_time:.3f}s")

        if block.end_time <= block.start_time:
            report.add_error(
                f"Cue #{idx} has zero or negative duration: {block.start_time:.3f}s --> {block.end_time:.3f}s"
            )

        # Chronological progression
        if idx > 1 and block.start_time < prev_start:
            report.add_error(
                f"Cue #{idx} starts earlier than preceding cue #{idx-1}: "
                f"({block.start_time:.3f}s < {prev_start:.3f}s)"
            )

        # Overlap check
        if idx > 1 and block.start_time < prev_end - 0.05:
            overlap = prev_end - block.start_time
            report.add_warning(
                f"Cue #{idx} overlaps with cue #{idx-1} by {overlap:.3f}s "
                f"(Cue #{idx-1} ends at {prev_end:.3f}s, Cue #{idx} starts at {block.start_time:.3f}s)"
            )

        # Media duration bound
        if video_duration is not None and block.end_time > video_duration + 0.5:
            overhang = block.end_time - video_duration
            report.add_warning(
                f"Cue #{idx} extends {overhang:.2f}s beyond media duration ({video_duration:.2f}s)"
            )

        prev_start = block.start_time
        prev_end = block.end_time

    return report
