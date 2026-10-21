"""
SRT Subtitle Serializer and Writer Module.

Converts SubtitleBlock objects into strict, CapCut-compatible SubRip (.srt) text format.
Guarantees precise milliseconds formatting: HH:MM:SS,mmm without drift or arbitrary rounding.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence
from speech_srt.sentence_segmenter import SubtitleBlock


def format_timestamp(seconds: float) -> str:
    """
    Format a floating-point seconds value into strict SRT timestamp syntax: HH:MM:SS,mmm.

    CapCut requires comma separator ',' between seconds and milliseconds, and exactly
    three digits for milliseconds.

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


def write_srt(blocks: Sequence[SubtitleBlock], output_path: Path) -> Path:
    """
    Write serialized SRT subtitles to destination file using UTF-8 encoding.

    Args:
        blocks: Sequence of SubtitleBlock objects.
        output_path: Path to destination .srt file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = serialize_srt(blocks)
    output_path.write_text(content, encoding="utf-8")
    return output_path
