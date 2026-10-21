"""
Media Information Inspection Module.

Inspects video and audio files using FFprobe without using shell=True,
extracting streams, audio codec, duration, sample rate, and channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MediaMetadata:
    """Represents essential technical metadata for an input media file."""
    file_path: Path
    duration_seconds: float
    has_audio: bool
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    video_codec: Optional[str] = None
    file_size_bytes: int = 0


def get_media_info(file_path: Path) -> MediaMetadata:
    """
    Extract media stream metadata using ffprobe.

    Args:
        file_path: Absolute or relative Path to video or audio file.

    Returns:
        MediaMetadata instance containing technical stream information.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If file is not a valid or readable media container.
        NotImplementedError: Implementation planned for Milestone 2.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Media file not found: {file_path}")

    # Core FFprobe inspection pipeline will be implemented in Milestone 2
    raise NotImplementedError(
        "Media inspection via FFprobe is scheduled for implementation in Milestone 2."
    )
