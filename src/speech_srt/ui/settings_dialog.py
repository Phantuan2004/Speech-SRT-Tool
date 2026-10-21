"""
Settings Dialog Component.

Allows user configuration of ASR engine, device selection (CPU / CUDA),
compute precision, and subtitle layout constraints.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QWidget
from speech_srt.config import AppConfig


class SettingsDialog(QDialog):
    """Configuration dialog for application parameters."""

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings -- Speech SRT Tool")
        self.resize(450, 320)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Settings configuration interface (Milestone 5)."))

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
