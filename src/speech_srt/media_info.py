"""
Media Information Inspection Module.

Inspects video and audio files using FFprobe without using shell=True,
extracting stream details, duration, sample rate, channels, and codecs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class MediaInspectionError(Exception):
    """Raised when media inspection fails or file is corrupt."""
    pass


class NoAudioStreamError(MediaInspectionError):
    """Raised when an inspected media file contains no audio streams."""
    pass


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


def get_media_info(file_path: Path, require_audio: bool = True) -> MediaMetadata:
    """
    Extract media stream metadata using ffprobe.

    Args:
        file_path: Path to video or audio file.
        require_audio: If True, raises NoAudioStreamError when audio stream is absent.

    Returns:
        MediaMetadata instance containing technical stream information.

    Raises:
        FileNotFoundError: If input file does not exist.
        MediaInspectionError: If ffprobe returns an error or fails to parse JSON.
        NoAudioStreamError: If file has no audio stream and require_audio is True.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Media file does not exist: {path}")
    if not path.is_file():
        raise MediaInspectionError(f"Path is not a regular file: {path}")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise MediaInspectionError(
            "ffprobe executable not found in system PATH. Please ensure FFmpeg is installed."
        )

    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaInspectionError(f"FFprobe inspection timed out for: {path.name}") from exc
    except Exception as exc:
        raise MediaInspectionError(f"Failed to execute FFprobe: {exc}") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or f"ffprobe exited with code {proc.returncode}"
        raise MediaInspectionError(f"Cannot inspect media file '{path.name}': {err_msg}")

    try:
        data: Dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaInspectionError(
            f"Failed to parse ffprobe JSON output for '{path.name}': {exc}"
        ) from exc

    streams: List[Dict[str, Any]] = data.get("streams", [])
    format_info: Dict[str, Any] = data.get("format", {})

    audio_stream: Optional[Dict[str, Any]] = None
    video_stream: Optional[Dict[str, Any]] = None

    for s in streams:
        c_type = s.get("codec_type")
        if c_type == "audio" and audio_stream is None:
            audio_stream = s
        elif c_type == "video" and video_stream is None:
            video_stream = s

    has_audio = audio_stream is not None

    if require_audio and not has_audio:
        raise NoAudioStreamError(
            f"No audio stream found in media file: '{path.name}'. Speech recognition requires an audio track."
        )

    # Determine duration: check format duration first, then stream duration
    duration = 0.0
    raw_duration = format_info.get("duration")
    if raw_duration:
        try:
            duration = float(raw_duration)
        except (ValueError, TypeError):
            pass

    if duration <= 0.0 and audio_stream:
        stream_dur = audio_stream.get("duration")
        if stream_dur:
            try:
                duration = float(stream_dur)
            except (ValueError, TypeError):
                pass

    if duration <= 0.0 and video_stream:
        stream_dur = video_stream.get("duration")
        if stream_dur:
            try:
                duration = float(stream_dur)
            except (ValueError, TypeError):
                pass

    # Extract audio details
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    sample_rate: Optional[int] = None
    if audio_stream and "sample_rate" in audio_stream:
        try:
            sample_rate = int(audio_stream["sample_rate"])
        except (ValueError, TypeError):
            pass

    channels: Optional[int] = None
    if audio_stream and "channels" in audio_stream:
        try:
            channels = int(audio_stream["channels"])
        except (ValueError, TypeError):
            pass

    video_codec = video_stream.get("codec_name") if video_stream else None

    file_size = 0
    try:
        file_size = path.stat().st_size
    except OSError:
        pass

    return MediaMetadata(
        file_path=path,
        duration_seconds=max(0.0, duration),
        has_audio=has_audio,
        audio_codec=audio_codec,
        sample_rate=sample_rate,
        channels=channels,
        video_codec=video_codec,
        file_size_bytes=file_size,
    )
