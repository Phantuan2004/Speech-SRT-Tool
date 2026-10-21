"""
Unit tests for ASR Engine, Device Resolution, and Lazy Loading.
"""

import pytest
from pathlib import Path
from speech_srt.asr_engine import (
    ASREngineError,
    DeviceUnavailableError,
    FasterWhisperEngine,
    ModelNotFoundError,
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
    is_cuda_supported,
    resolve_device_and_compute_type,
)


class TestASREngineBasics:
    """Tests for device resolution, typing, and lazy loading."""

    def test_cuda_detection_on_current_machine(self) -> None:
        # Current machine has AMD GPU, no CUDA
        assert is_cuda_supported() is False

    def test_device_resolution_auto_defaults_to_cpu(self) -> None:
        dev, comp = resolve_device_and_compute_type("auto")
        assert dev == "cpu"
        assert comp == "int8"

    def test_device_resolution_cpu_explicit(self) -> None:
        dev, comp = resolve_device_and_compute_type("cpu", "float32")
        assert dev == "cpu"
        assert comp == "float32"

    def test_device_resolution_cuda_when_unavailable_raises_error(self) -> None:
        with pytest.raises(DeviceUnavailableError, match="no CUDA-capable device"):
            resolve_device_and_compute_type("cuda")

    def test_invalid_device_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            resolve_device_and_compute_type("tpu")

    def test_lazy_loading_does_not_load_at_init(self) -> None:
        engine = FasterWhisperEngine(model_size="small.en", device="cpu")
        assert engine._is_loaded is False
        assert engine._model is None

    def test_offline_mode_with_uncached_model_raises_model_not_found(self, tmp_path: Path) -> None:
        # Request valid model size in an empty local directory with offline=True
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir(parents=True, exist_ok=True)
        engine = FasterWhisperEngine(
            model_size="small.en",
            device="cpu",
            download_root=empty_cache,
        )
        with pytest.raises(ModelNotFoundError, match="not available in local cache"):
            engine.load_model(offline=True)

    def test_invalid_model_size_raises_asr_engine_error(self) -> None:
        engine = FasterWhisperEngine(model_size="invalid_size_xyz")
        with pytest.raises(ASREngineError, match="Invalid model size"):
            engine.load_model()

    def test_transcription_result_all_words_property(self) -> None:
        words1 = [WordTiming(word="Hello", start=0.0, end=0.5)]
        words2 = [WordTiming(word="World", start=0.6, end=1.0)]

        seg1 = TranscriptionSegment(id=1, seek=0, start=0.0, end=0.5, text="Hello", words=words1)
        seg2 = TranscriptionSegment(id=2, seek=0, start=0.6, end=1.0, text="World", words=words2)

        result = TranscriptionResult(language="en", duration=1.0, segments=[seg1, seg2])
        all_words = result.all_words

        assert len(all_words) == 2
        assert all_words[0].word == "Hello"
        assert all_words[1].word == "World"
