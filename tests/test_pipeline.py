"""
Unit tests for Pipeline Orchestrator baseline setup.
"""

from pathlib import Path
from speech_srt.config import AppConfig, load_config
from speech_srt.pipeline import TranscriptionPipeline


class TestPipelineBaseline:
    """Tests for TranscriptionPipeline configuration and initial validation."""

    def test_pipeline_initialization_with_defaults(self) -> None:
        pipeline = TranscriptionPipeline()
        assert pipeline.config is not None
        assert pipeline.config.target_language == "en"
        assert pipeline.config.asr.engine == "faster-whisper"
        assert pipeline.config.asr.device == "cpu"

    def test_pipeline_custom_config(self) -> None:
        custom_cfg = AppConfig(target_language="en")
        pipeline = TranscriptionPipeline(config=custom_cfg)
        assert pipeline.config.target_language == "en"

    def test_pipeline_rejects_missing_file(self, tmp_path: Path) -> None:
        pipeline = TranscriptionPipeline()
        missing_video = tmp_path / "non_existent.mp4"
        result = pipeline.process(missing_video)

        assert not result.success
        assert result.output_srt_file is None
        assert "validation failed" in result.error_message.lower()
