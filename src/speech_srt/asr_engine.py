"""
Automatic Speech Recognition (ASR) Engine Interface & Faster-Whisper Implementation.

Defines standardized data structures and provides lazy loading for the
faster-whisper transcription engine. Supports CPU (int8) and NVIDIA CUDA (float16)
with automatic hardware fallback and word-level timestamp extraction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ASREngineError(Exception):
    """Base exception for ASR engine failures."""
    pass


class ModelNotFoundError(ASREngineError):
    """Raised when a model is requested offline but has not been downloaded or cached."""
    pass


class DeviceUnavailableError(ASREngineError):
    """Raised when an explicitly requested hardware device (e.g. CUDA) is not available."""
    pass


@dataclass
class WordTiming:
    """Represents a recognized word with exact start and end timestamps."""
    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionSegment:
    """Represents an ASR transcription segment with timing and contained words."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Normalized output from ASR transcription."""
    language: str
    duration: float
    segments: List[TranscriptionSegment] = field(default_factory=list)

    @property
    def all_words(self) -> List[WordTiming]:
        words = []
        for segment in self.segments:
            words.extend(segment.words)
        return words


def is_cuda_supported() -> bool:
    """Check if CUDA runtime and compatible GPU are available for ctranslate2."""
    try:
        import ctranslate2
        if hasattr(ctranslate2, "get_cuda_device_count"):
            return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    return False


def resolve_device_and_compute_type(
    requested_device: str = "auto",
    requested_compute_type: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve target device ('cpu' or 'cuda') and compute precision safely.

    Args:
        requested_device: 'auto', 'cpu', or 'cuda'.
        requested_compute_type: Optional explicit compute type (e.g. 'int8', 'float16').

    Returns:
        Tuple of (resolved_device, resolved_compute_type).

    Raises:
        DeviceUnavailableError: If 'cuda' is explicitly requested but unavailable.
    """
    req_dev = requested_device.lower().strip()

    if req_dev == "cuda":
        if not is_cuda_supported():
            raise DeviceUnavailableError(
                "CUDA was requested, but no CUDA-capable device or CUDA runtime was detected."
            )
        resolved_device = "cuda"
        resolved_compute = requested_compute_type or "float16"
    elif req_dev == "cpu":
        resolved_device = "cpu"
        resolved_compute = requested_compute_type or "int8"
    elif req_dev == "auto":
        if is_cuda_supported():
            resolved_device = "cuda"
            resolved_compute = requested_compute_type or "float16"
        else:
            resolved_device = "cpu"
            resolved_compute = requested_compute_type or "int8"
    else:
        raise ValueError(f"Unsupported device: '{requested_device}'. Choose 'auto', 'cpu', or 'cuda'.")

    return resolved_device, resolved_compute


class BaseASREngine(ABC):
    """Abstract base class for speech recognition engines."""

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[Path] = None,
    ) -> None:
        self.model_size = model_size
        self.device, self.compute_type = resolve_device_and_compute_type(device, compute_type)
        self.download_root = download_root
        self._is_loaded = False

    @abstractmethod
    def load_model(self, offline: bool = False) -> None:
        """Lazily load or download model. Never execute during module import."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an audio file and return normalized segments and word timings."""
        raise NotImplementedError


class FasterWhisperEngine(BaseASREngine):
    """
    Offline English ASR using faster-whisper (CTranslate2).
    Model weights are loaded lazily on demand.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[Path] = None,
        cpu_threads: int = 4,
    ) -> None:
        super().__init__(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        self.cpu_threads = cpu_threads
        self._model = None

    def load_model(self, offline: bool = False) -> None:
        """
        Load Whisper model into memory.

        Args:
            offline: If True, restricts search to local cache without network downloads.

        Raises:
            ModelNotFoundError: If model files are missing in offline mode.
            ASREngineError: If initialization fails.
        """
        if self._is_loaded and self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASREngineError(
                "faster-whisper is not installed. Please install faster-whisper>=1.0.0."
            ) from exc

        root_str = str(self.download_root) if self.download_root else None

        try:
            self._model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=root_str,
                local_files_only=offline,
            )
            self._is_loaded = True
        except Exception as exc:
            err_text = str(exc).lower()
            if (
                "local_files_only" in err_text
                or "cannot find" in err_text
                or "local disk" in err_text
                or "offline" in err_text
                or "not found" in err_text
            ):
                raise ModelNotFoundError(
                    f"Model '{self.model_size}' is not available in local cache. "
                    f"Run without offline restriction to allow downloading model weights."
                ) from exc
            if "invalid model size" in err_text:
                raise ASREngineError(f"Invalid model size '{self.model_size}': {exc}") from exc
            raise ASREngineError(
                f"Failed to load WhisperModel '{self.model_size}' on {self.device} ({self.compute_type}): {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: Path,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio file into English text with word-level timestamps.

        Args:
            audio_path: Path to 16kHz mono WAV file.
            beam_size: Beam search width (default: 5).
            vad_filter: Silero Voice Activity Detector filter (default: True).
            word_timestamps: Extract word-level start/end timings (default: True).

        Returns:
            TranscriptionResult containing segments and word timings.

        Raises:
            FileNotFoundError: If audio file does not exist.
            ASREngineError: If transcription fails.
        """
        audio = Path(audio_path).resolve()
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found for transcription: {audio}")

        # Ensure model is initialized
        if not self._is_loaded or self._model is None:
            self.load_model()

        try:
            segments_gen, info = self._model.transcribe(
                str(audio),
                language="en",
                task="transcribe",
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                condition_on_previous_text=False,  # Prevents repetition loops in long speech
            )

            result_segments: List[TranscriptionSegment] = []
            for seg in segments_gen:
                seg_words: List[WordTiming] = []
                if seg.words:
                    for w in seg.words:
                        # Retain exact word without stripping filler syllables
                        seg_words.append(
                            WordTiming(
                                word=w.word,
                                start=w.start,
                                end=w.end,
                                probability=getattr(w, "probability", 1.0),
                            )
                        )

                result_segments.append(
                    TranscriptionSegment(
                        id=seg.id,
                        seek=seg.seek,
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                        words=seg_words,
                    )
                )

            return TranscriptionResult(
                language=info.language,
                duration=info.duration,
                segments=result_segments,
            )
        except Exception as exc:
            raise ASREngineError(f"Transcription failed for '{audio.name}': {exc}") from exc
