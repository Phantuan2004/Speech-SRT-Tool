"""
Validation Utilities Module.

Validates input files, timestamp coherence, duration bounds, and SRT syntax.
Ensures zero data corruption and detects invalid inputs before processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple
from speech_srt.sentence_segmenter import SubtitleBlock

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"
}
SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"
}
SUPPORTED_EXTENSIONS: Set[str] = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS

# Regex for strict SRT timestamp line: 00:00:00,000 --> 00:00:00,000
SRT_TIMESTAMP_REGEX = re.compile(
    r"^\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)


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
    Validate that start and end timestamps form a valid, non-negative, forward-progressing interval.

    Args:
        start_time: Interval start in seconds.
        end_time: Interval end in seconds.
        max_duration: Optional upper bound for end_time (e.g. media duration).

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if start_time < 0.0:
        return False, f"Start time cannot be negative: {start_time}"

    if end_time < 0.0:
        return False, f"End time cannot be negative: {end_time}"

    if end_time <= start_time:
        return False, f"End time ({end_time}s) must be strictly greater than start time ({start_time}s)"

    if max_duration is not None and end_time > max_duration + 0.5:
        return False, f"End time ({end_time}s) exceeds media duration ({max_duration}s)"

    return True, "Valid timestamp range"


def validate_subtitle_blocks(blocks: Sequence[SubtitleBlock]) -> Tuple[bool, str]:
    """
    Validate an ordered sequence of SubtitleBlock items.
    Checks for index continuity, valid timestamps, and non-empty text.

    Args:
        blocks: Sequence of SubtitleBlock instances.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if not blocks:
        return True, "Empty subtitle sequence"

    for idx, block in enumerate(blocks, start=1):
        if block.index != idx:
            return False, f"Subtitle block index mismatch: expected {idx}, got {block.index}"

        is_valid, err = validate_timestamps(block.start_time, block.end_time)
        if not is_valid:
            return False, f"Block #{idx} timestamp error: {err}"

        if not block.text.strip():
            return False, f"Block #{idx} has empty text"

    return True, "All subtitle blocks are valid"
