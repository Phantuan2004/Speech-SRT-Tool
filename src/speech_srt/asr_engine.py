"""
Automatic Speech Recognition (ASR) Engine Interface.

Defines data contracts and base class for offline speech recognition.
Models are lazily loaded on-demand; no weights are downloaded or loaded at import time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class WordTiming:
    """Represents a single recognized word with start and end timestamps."""
    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionSegment:
    """Represents a spoken phrase or sentence segment with timing information."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete result returned from ASR transcription."""
    language: str
    duration: float
    segments: List[TranscriptionSegment] = field(default_factory=list)

    @property
    def all_words(self) -> List[WordTiming]:
        words = []
        for segment in self.segments:
            words.extend(segment.words)
        return words


class BaseASREngine(ABC):
    """Abstract base class for all ASR engines."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[Path] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Lazily load model weights into memory. Never execute at module import."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file and return word-aligned transcription segments."""
        raise NotImplementedError


class FasterWhisperEngine(BaseASREngine):
    """Offline Faster-Whisper ASR implementation with lazy loading."""

    def load_model(self) -> None:
        # Implementation scheduled for Milestone 3
        raise NotImplementedError(
            "FasterWhisperEngine model loading is scheduled for implementation in Milestone 3."
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        # Implementation scheduled for Milestone 3
        raise NotImplementedError(
            "Transcription pipeline is scheduled for implementation in Milestone 3."
        )
