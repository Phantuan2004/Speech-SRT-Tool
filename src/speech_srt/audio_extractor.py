"""
Audio Extraction Module.

Extracts normalized 16kHz mono PCM WAV audio from input video files
using FFmpeg with direct argument tokenization (no shell=True).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def extract_audio(
    input_video: Path,
    output_audio: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Extract audio track from video file and convert to 16kHz mono WAV.

    Args:
        input_video: Path to source video file.
        output_audio: Optional target Path for the extracted WAV audio.
        sample_rate: Target audio sampling rate in Hz (default: 16000 for Whisper).
        channels: Target audio channel count (default: 1 mono).

    Returns:
        Path to the extracted WAV audio file.

    Raises:
        FileNotFoundError: If input video does not exist.
        NotImplementedError: Implementation planned for Milestone 2.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    # Core FFmpeg audio extraction pipeline will be implemented in Milestone 2
    raise NotImplementedError(
        "Audio extraction via FFmpeg is scheduled for implementation in Milestone 2."
    )
