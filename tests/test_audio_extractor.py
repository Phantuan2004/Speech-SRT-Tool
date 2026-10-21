"""
Unit tests for Audio Extraction Module (FFmpeg).
"""

import subprocess
import pytest
from pathlib import Path
from speech_srt.audio_extractor import (
    AudioExtractionError,
    extract_audio,
    extract_audio_temp,
)
from speech_srt.media_info import get_media_info


@pytest.fixture(scope="module")
def input_test_video(tmp_path_factory) -> Path:
    """Generate a 2-second MP4 video containing video and audio streams."""
    tmp_dir = tmp_path_factory.mktemp("audio_extraction_fixtures")
    video_path = tmp_dir / "test_video_with_audio.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-c:a", "aac",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


class TestAudioExtractor:
    """Tests for extract_audio and extract_audio_temp."""

    def test_extract_audio_creates_16k_mono_wav(self, input_test_video: Path, tmp_path: Path) -> None:
        target_wav = tmp_path / "extracted.wav"
        result_path = extract_audio(input_test_video, output_audio=target_wav)

        assert result_path == target_wav
        assert target_wav.exists()

        # Verify stream specs via media_info
        info = get_media_info(target_wav)
        assert info.has_audio is True
        assert info.channels == 1
        assert info.sample_rate == 16000
        assert info.audio_codec == "pcm_s16le"
        assert 1.8 <= info.duration_seconds <= 2.2

    def test_extract_audio_temp_guarantees_cleanup(self, input_test_video: Path) -> None:
        captured_path: Path = None
        with extract_audio_temp(input_test_video) as temp_wav:
            captured_path = temp_wav
            assert captured_path.exists()
            assert captured_path.stat().st_size > 0

        # After exiting context manager, temp file must be cleaned up
        assert not captured_path.exists()

    def test_same_input_and_output_raises_error(self, input_test_video: Path) -> None:
        with pytest.raises(AudioExtractionError, match="cannot be the same"):
            extract_audio(input_test_video, output_audio=input_test_video)

    def test_missing_input_raises_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.mp4"
        with pytest.raises(FileNotFoundError):
            extract_audio(missing)
