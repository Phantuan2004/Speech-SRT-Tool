"""
Configuration Management Module.

Loads, validates, and manages application settings using typed dataclasses
and pathlib for cross-platform compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ASRConfig:
    engine: str = "faster-whisper"
    model_size: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    offline_mode: bool = True


@dataclass
class SubtitleConfig:
    format: str = "srt"
    capcut_compatible: bool = True
    max_line_length: int = 42
    max_lines_per_subtitle: int = 2
    target_accuracy_tolerance_ms: int = 200
    target_accuracy_percentage: float = 95.0


@dataclass
class AudioExtractionConfig:
    sample_rate: int = 16000
    channels: int = 1
    codec: str = "pcm_s16le"
    format: str = "wav"


@dataclass
class StorageConfig:
    output_dir: str = "output"
    log_dir: str = "logs"


@dataclass
class AppConfig:
    app_name: str = "Speech SRT Tool"
    version: str = "0.1.0"
    target_language: str = "en"
    asr: ASRConfig = field(default_factory=ASRConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    audio_extraction: AudioExtractionConfig = field(default_factory=AudioExtractionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def resolve_output_dir(self, base_path: Optional[Path] = None) -> Path:
        """Resolve output directory to an absolute Path."""
        base = base_path or Path.cwd()
        return (base / self.storage.output_dir).resolve()

    def resolve_log_dir(self, base_path: Optional[Path] = None) -> Path:
        """Resolve log directory to an absolute Path."""
        base = base_path or Path.cwd()
        return (base / self.storage.log_dir).resolve()


def get_default_config_path() -> Path:
    """Return default configuration path relative to repository root."""
    return Path(__file__).resolve().parent.parent.parent / "config" / "default_settings.json"


def load_config(config_file: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from JSON file or return default configuration if file is missing.

    Args:
        config_file: Optional path to JSON settings file.

    Returns:
        AppConfig: Populated typed configuration instance.
    """
    target_path = config_file or get_default_config_path()

    if not target_path.exists():
        return AppConfig()

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        return AppConfig(
            app_name=data.get("app_name", "Speech SRT Tool"),
            version=data.get("version", "0.1.0"),
            target_language=data.get("target_language", "en"),
            asr=ASRConfig(**data.get("asr", {})),
            subtitles=SubtitleConfig(**data.get("subtitles", {})),
            audio_extraction=AudioExtractionConfig(**data.get("audio_extraction", {})),
            storage=StorageConfig(**data.get("storage", {})),
        )
    except Exception:
        # Fallback to robust defaults if config is corrupted
        return AppConfig()
