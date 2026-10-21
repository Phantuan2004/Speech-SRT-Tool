"""
Main Application Window.

PySide6 desktop interface featuring drag-and-drop video loading,
real-time transcription progress tracking, and subtitle preview table.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from speech_srt.config import AppConfig


class MainWindow(QMainWindow):
    """Main application window for Speech SRT Tool."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Speech SRT Tool -- English Subtitle Generator")
        self.resize(960, 640)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        status_label = QLabel("Speech SRT Tool Desktop Interface (Milestone 5 GUI)")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)
