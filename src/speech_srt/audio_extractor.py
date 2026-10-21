"""
Audio Extraction Module.

Extracts normalized 16kHz mono PCM WAV audio from input video or audio files
using FFmpeg without shell=True, preserving exact media duration and playback rate.
Provides isolated temporary directory handling and reliable cleanup on failures.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional


class AudioExtractionError(Exception):
    """Raised when FFmpeg audio extraction fails."""
    pass


def extract_audio(
    input_video: Path,
    output_audio: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Extract audio track from video/audio and convert to 16kHz mono WAV PCM.

    Args:
        input_video: Source media path.
        output_audio: Target path for extracted WAV. If None, creates a file in a temporary folder.
        sample_rate: Target sampling rate in Hz (default: 16000).
        channels: Target channel count (default: 1 mono).

    Returns:
        Path to the extracted WAV audio file.

    Raises:
        FileNotFoundError: If input file does not exist.
        AudioExtractionError: If extraction fails or destination clashes with source.
    """
    src_path = Path(input_video).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Input media does not exist: {src_path}")
    if not src_path.is_file():
        raise AudioExtractionError(f"Input media is not a regular file: {src_path}")

    # Determine destination
    if output_audio is not None:
        dst_path = Path(output_audio).resolve()
    else:
        task_id = uuid.uuid4().hex[:8]
        temp_dir = Path(tempfile.gettempdir()) / "speech_srt_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        dst_path = temp_dir / f"{src_path.stem}_{task_id}_16k_mono.wav"

    # Safety: Never overwrite source file
    if dst_path == src_path:
        raise AudioExtractionError(
            f"Output audio path cannot be the same as input source file: {src_path}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise AudioExtractionError(
            "ffmpeg executable not found in system PATH. Please ensure FFmpeg is installed."
        )

    # Command token list: strictly no shell=True
    cmd = [
        ffmpeg_bin,
        "-y",               # Overwrite destination if temporary leftover exists
        "-v", "error",      # Suppress verbose banners, output errors only
        "-i", str(src_path),
        "-vn",              # Disable video recording
        "-acodec", "pcm_s16le",  # Uncompressed 16-bit PCM WAV
        "-ar", str(sample_rate), # 16000 Hz
        "-ac", str(channels),    # Mono channel
        str(dst_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        _safe_cleanup(dst_path)
        raise AudioExtractionError(
            f"Audio extraction timed out for file: '{src_path.name}'"
        ) from exc
    except Exception as exc:
        _safe_cleanup(dst_path)
        raise AudioExtractionError(f"Failed to execute FFmpeg: {exc}") from exc

    if proc.returncode != 0:
        _safe_cleanup(dst_path)
        err_msg = proc.stderr.strip() or f"FFmpeg exited with error code {proc.returncode}"
        raise AudioExtractionError(f"FFmpeg audio extraction failed for '{src_path.name}': {err_msg}")

    if not dst_path.exists() or dst_path.stat().st_size == 0:
        _safe_cleanup(dst_path)
        raise AudioExtractionError(
            f"Extracted audio file is missing or 0 bytes: {dst_path}"
        )

    return dst_path


@contextmanager
def extract_audio_temp(
    input_video: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Generator[Path, None, None]:
    """
    Context manager that extracts 16kHz mono audio to an isolated temporary file,
    yielding its Path and guaranteeing safe cleanup upon exit or failure.

    Args:
        input_video: Source media path.
        sample_rate: Target sampling rate in Hz (default: 16000).
        channels: Target channels count (default: 1).

    Yields:
        Path to temporary extracted WAV file.
    """
    extracted_path: Optional[Path] = None
    try:
        extracted_path = extract_audio(
            input_video=input_video,
            output_audio=None,
            sample_rate=sample_rate,
            channels=channels,
        )
        yield extracted_path
    finally:
        if extracted_path is not None:
            _safe_cleanup(extracted_path)


def _safe_cleanup(path: Path) -> None:
    """Silently and safely remove a file if it exists."""
    try:
        if path.exists():
            path.unlink(missing_ok=True)
    except OSError:
        pass
