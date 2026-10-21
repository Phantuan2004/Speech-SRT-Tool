"""
Unit tests for Media Inspection Module (FFprobe).
"""

import subprocess
import pytest
from pathlib import Path
from speech_srt.media_info import (
    MediaInspectionError,
    NoAudioStreamError,
    get_media_info,
)


@pytest.fixture(scope="module")
def sample_audio_file(tmp_path_factory) -> Path:
    """Generate a clean 2-second 44.1kHz stereo WAV file using FFmpeg."""
    tmp_dir = tmp_path_factory.mktemp("media_fixtures")
    audio_path = tmp_dir / "sample_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-ar", "44100", "-ac", "2",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return audio_path


@pytest.fixture(scope="module")
def sample_video_no_audio(tmp_path_factory) -> Path:
    """Generate a 1-second MP4 video containing video only (no audio stream)."""
    tmp_dir = tmp_path_factory.mktemp("media_fixtures")
    video_path = tmp_dir / "no_audio.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-c:v", "libx264", "-an",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


class TestMediaInfo:
    """Tests for get_media_info function."""

    def test_nonexistent_file_raises_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_media_info(missing)

    def test_corrupt_file_raises_media_inspection_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_text("not a video at all", encoding="utf-8")
        with pytest.raises(MediaInspectionError):
            get_media_info(corrupt)

    def test_valid_audio_file_inspection(self, sample_audio_file: Path) -> None:
        info = get_media_info(sample_audio_file)
        assert info.has_audio is True
        assert info.channels == 2
        assert info.sample_rate == 44100
        assert 1.8 <= info.duration_seconds <= 2.2
        assert info.file_size_bytes > 0

    def test_video_without_audio_raises_no_audio_stream_error(self, sample_video_no_audio: Path) -> None:
        with pytest.raises(NoAudioStreamError, match="No audio stream found"):
            get_media_info(sample_video_no_audio, require_audio=True)

    def test_video_without_audio_when_not_required(self, sample_video_no_audio: Path) -> None:
        info = get_media_info(sample_video_no_audio, require_audio=False)
        assert info.has_audio is False
        assert info.video_codec is not None
        assert info.duration_seconds >= 0.9
