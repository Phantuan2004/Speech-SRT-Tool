"""
Subtitle Table Model and View Component.

Provides editable QAbstractTableModel for inspecting and fine-tuning
subtitles directly within the PySide6 desktop interface.
"""

from __future__ import annotations

from typing import Any, List, Optional
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from speech_srt.sentence_segmenter import SubtitleBlock


class SubtitleTableModel(QAbstractTableModel):
    """Table model representing a list of SubtitleBlock entries for QTableView."""

    COLUMNS = ["#", "Start Time", "End Time", "Subtitle Text"]

    def __init__(self, subtitles: Optional[List[SubtitleBlock]] = None) -> None:
        super().__init__()
        self._subtitles: List[SubtitleBlock] = subtitles or []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._subtitles)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._subtitles)):
            return None

        sub = self._subtitles[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return sub.index
            elif col == 1:
                return f"{sub.start_time:.3f}s"
            elif col == 2:
                return f"{sub.end_time:.3f}s"
            elif col == 3:
                return sub.text

        return None
