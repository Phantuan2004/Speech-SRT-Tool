"""
SRT Subtitle Serializer, Writer, and Parser Module.

Converts SubtitleBlock objects into strict, CapCut-compatible SubRip (.srt) format.
Provides atomic file writing to prevent corrupted outputs and round-trip SRT parsing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Sequence, Union
from speech_srt.sentence_segmenter import SubtitleBlock

SRT_TIMESTAMP_REGEX = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})$"
)


class SRTFormatError(Exception):
    """Raised when an SRT file has invalid formatting or corrupted syntax."""
    pass


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into strict CapCut SRT timestamp syntax: HH:MM:SS,mmm.

    Args:
        seconds: Time offset in seconds (e.g., 65.432). Must be non-negative.

    Returns:
        Formatted string, e.g., '00:01:05,432'.

    Raises:
        ValueError: If seconds is negative.
    """
    if seconds < 0:
        raise ValueError(f"Timestamp seconds cannot be negative, got: {seconds}")

    total_ms = int(round(seconds * 1000.0))

    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000

    minutes = remainder // 60_000
    remainder = remainder % 60_000

    secs = remainder // 1_000
    ms = remainder % 1_000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse an SRT timestamp (HH:MM:SS,mmm) back to fractional seconds.

    Args:
        timestamp_str: String formatted as '00:01:05,432'.

    Returns:
        Float value in seconds.

    Raises:
        SRTFormatError: If timestamp format is invalid.
    """
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", timestamp_str.strip())
    if not match:
        raise SRTFormatError(f"Invalid SRT timestamp syntax: '{timestamp_str}'")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def serialize_srt(blocks: Sequence[SubtitleBlock]) -> str:
    """
    Serialize an ordered sequence of SubtitleBlock instances into standard SRT format.

    Args:
        blocks: Sequence of SubtitleBlock objects.

    Returns:
        Complete UTF-8 compatible SRT file text content.
    """
    if not blocks:
        return ""

    entries: List[str] = []
    for idx, block in enumerate(blocks, start=1):
        start_str = format_timestamp(block.start_time)
        end_str = format_timestamp(block.end_time)
        text = block.text.strip()
        entries.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")

    return "\n".join(entries).strip() + "\n"


def parse_srt(content_or_path: Union[str, Path]) -> List[SubtitleBlock]:
    """
    Parse an SRT formatted string or file into a list of SubtitleBlock objects.

    Args:
        content_or_path: Raw SRT string content or Path to .srt file.

    Returns:
        List of parsed SubtitleBlock instances.

    Raises:
        FileNotFoundError: If path does not exist.
        SRTFormatError: If SRT syntax is invalid.
    """
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()):
        file_path = Path(content_or_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SRT file not found: {file_path}")
        text = file_path.read_text(encoding="utf-8")
    else:
        text = str(content_or_path)

    # Normalize newlines
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    # Split by double newline (standard subtitle block separator)
    raw_blocks = re.split(r"\n\s*\n", normalized)
    blocks: List[SubtitleBlock] = []

    for block_str in raw_blocks:
        lines = block_str.strip().splitlines()
        if len(lines) < 3:
            raise SRTFormatError(f"Malformed SRT block with fewer than 3 lines: {repr(block_str)}")

        # Line 1: Index
        try:
            index = int(lines[0].strip())
        except ValueError as exc:
            raise SRTFormatError(f"Invalid subtitle index '{lines[0]}'") from exc

        # Line 2: Timestamp arrow
        arrow_match = SRT_TIMESTAMP_REGEX.match(lines[1].strip())
        if not arrow_match:
            raise SRTFormatError(f"Invalid timestamp line in block #{index}: '{lines[1]}'")

        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, arrow_match.groups())
        start_time = h1 * 3600.0 + m1 * 60.0 + s1 + ms1 / 1000.0
        end_time = h2 * 3600.0 + m2 * 60.0 + s2 + ms2 / 1000.0

        if end_time <= start_time:
            raise SRTFormatError(
                f"Block #{index} has invalid interval: start ({start_time}s) >= end ({end_time}s)"
            )

        # Remaining lines: Subtitle text
        cue_text = "\n".join(lines[2:]).strip()
        if not cue_text:
            raise SRTFormatError(f"Block #{index} contains empty subtitle text")

        blocks.append(
            SubtitleBlock(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=cue_text,
            )
        )

    return blocks


def write_srt(blocks: Sequence[SubtitleBlock], output_path: Path) -> Path:
    """
    Atomically write serialized SRT subtitles to target path using UTF-8 encoding.
    Uses a temporary sibling file and atomic replacement to prevent corrupting existing files on crash.

    Args:
        blocks: Sequence of SubtitleBlock objects.
        output_path: Path to destination .srt file.

    Returns:
        Path to the written file.

    Raises:
        OSError: If file writing or renaming fails.
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = serialize_srt(blocks)

    # Atomic write pattern: write to .tmp then os.replace
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return out_path
