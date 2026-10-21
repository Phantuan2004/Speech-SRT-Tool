"""
Background Worker Threads for Asynchronous Processing.

Keeps PySide6 GUI responsive by running FFmpeg and ASR pipelines on background threads.
Emits Qt signals for progress percentage, status updates, completion, and error states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Signal
from speech_srt.config import AppConfig


class TranscriptionWorker(QObject):
    """Worker object to run transcription pipeline asynchronously inside a QThread."""

    # Qt Signals
    progress_changed = Signal(float, str)  # percentage (0.0 - 100.0), message
    transcription_finished = Signal(str)   # output srt file path
    error_occurred = Signal(str)           # error description

    def __init__(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self._is_cancelled = False

    def cancel(self) -> None:
        """Flag worker to abort processing safely."""
        self._is_cancelled = True

    def run(self) -> None:
        """Main execution loop dispatched by the thread."""
        # Full worker execution will be implemented in Milestone 5
        self.progress_changed.emit(0.0, "Worker initialized")
