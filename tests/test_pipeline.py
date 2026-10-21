"""
Unit and Integration tests for Transcription Pipeline.
"""

import subprocess
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from speech_srt.asr_engine import TranscriptionResult, TranscriptionSegment, WordTiming
from speech_srt.pipeline import TranscriptionPipeline
from speech_srt.srt_writer import parse_srt


@pytest.fixture(scope="module")
def fixture_video_with_audio(tmp_path_factory) -> Path:
    """Generate a clean 2-second test MP4 with AAC audio stream."""
    tmp_dir = tmp_path_factory.mktemp("pipeline_fixtures")
    video_path = tmp_dir / "sample_video.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-c:a", "aac",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


@pytest.fixture(scope="module")
def fixture_video_without_audio(tmp_path_factory) -> Path:
    """Generate a 1-second video without any audio stream."""
    tmp_dir = tmp_path_factory.mktemp("pipeline_fixtures")
    video_path = tmp_dir / "no_audio.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-c:v", "libx264", "-an",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


class TestPipelineOrchestration:
    """Tests for TranscriptionPipeline execution flows."""

    def test_pipeline_rejects_missing_file(self, tmp_path: Path) -> None:
        pipeline = TranscriptionPipeline()
        missing = tmp_path / "not_there.mp4"
        result = pipeline.process(missing)
        assert not result.success
        assert "validation error" in result.error_message.lower()

    def test_pipeline_fails_on_video_without_audio(self, fixture_video_without_audio: Path) -> None:
        pipeline = TranscriptionPipeline()
        result = pipeline.process(fixture_video_without_audio)
        assert not result.success
        assert "No audio stream found" in result.error_message

    def test_pipeline_runs_with_mocked_asr(
        self,
        fixture_video_with_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_srt = tmp_path / "output.srt"
        pipeline = TranscriptionPipeline()

        # Mock ASR engine to test the entire media extraction, segmentation, validation, and SRT write
        mock_words = [
            WordTiming(word="Testing", start=0.2, end=0.7),
            WordTiming(word="subtitles", start=0.8, end=1.3),
            WordTiming(word="pipeline.", start=1.4, end=1.8),
        ]
        mock_seg = TranscriptionSegment(
            id=1,
            seek=0,
            start=0.2,
            end=1.8,
            text="Testing subtitles pipeline.",
            words=mock_words,
        )
        mock_transcription = TranscriptionResult(
            language="en",
            duration=2.0,
            segments=[mock_seg],
        )

        mock_engine = MagicMock()
        mock_engine.device = "cpu"
        mock_engine.compute_type = "int8"
        mock_engine.transcribe.return_value = mock_transcription
        pipeline._engine = mock_engine

        progress_events = []
        def on_progress(pct: float, msg: str) -> None:
            progress_events.append((pct, msg))

        result = pipeline.process(
            input_file=fixture_video_with_audio,
            output_srt_file=output_srt,
            progress_callback=on_progress,
        )

        assert result.success is True
        assert result.output_srt_file == output_srt
        assert output_srt.exists()
        assert result.subtitle_count == 1
        assert len(progress_events) >= 5

        # Verify generated SRT is parseable and content is correct
        parsed = parse_srt(output_srt)
        assert len(parsed) == 1
        assert parsed[0].text == "Testing subtitles pipeline."
        assert abs(parsed[0].start_time - 0.2) < 1e-3
        assert abs(parsed[0].end_time - 1.8) < 1e-3
