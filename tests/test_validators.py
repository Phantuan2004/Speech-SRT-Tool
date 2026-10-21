"""
Unit tests for Validator Utilities.
"""

from pathlib import Path
from speech_srt.sentence_segmenter import SubtitleBlock
from speech_srt.validators import (
    validate_media_file,
    validate_timestamps,
    validate_subtitle_blocks,
)


class TestValidateMediaFile:
    """Tests for validate_media_file function."""

    def test_nonexistent_file_fails(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "missing_video.mp4"
        is_valid, err = validate_media_file(missing_file)
        assert not is_valid
        assert "does not exist" in err

    def test_empty_file_fails(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty_video.mp4"
        empty_file.touch()
        is_valid, err = validate_media_file(empty_file)
        assert not is_valid
        assert "empty (0 bytes)" in err

    def test_unsupported_extension_fails(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("dummy content", encoding="utf-8")
        is_valid, err = validate_media_file(text_file)
        assert not is_valid
        assert "Unsupported file extension" in err

    def test_valid_media_file_passes(self, tmp_path: Path) -> None:
        valid_video = tmp_path / "sample.mp4"
        valid_video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        is_valid, err = validate_media_file(valid_video)
        assert is_valid
        assert "Valid media file" in err


class TestValidateTimestamps:
    """Tests for validate_timestamps function."""

    def test_negative_start_fails(self) -> None:
        is_valid, err = validate_timestamps(-0.5, 2.0)
        assert not is_valid
        assert "Start time cannot be negative" in err

    def test_negative_end_fails(self) -> None:
        is_valid, err = validate_timestamps(0.0, -1.0)
        assert not is_valid
        assert "End time cannot be negative" in err

    def test_end_before_or_equal_to_start_fails(self) -> None:
        is_valid, err = validate_timestamps(2.0, 1.5)
        assert not is_valid
        assert "strictly greater" in err

        is_valid, err = validate_timestamps(2.0, 2.0)
        assert not is_valid
        assert "strictly greater" in err

    def test_end_exceeds_max_duration_fails(self) -> None:
        is_valid, err = validate_timestamps(10.0, 25.0, max_duration=20.0)
        assert not is_valid
        assert "exceeds media duration" in err

    def test_valid_timestamps_pass(self) -> None:
        is_valid, err = validate_timestamps(1.0, 4.5, max_duration=10.0)
        assert is_valid
        assert "Valid timestamp range" in err


class TestValidateSubtitleBlocks:
    """Tests for validate_subtitle_blocks function."""

    def test_empty_sequence_is_valid(self) -> None:
        is_valid, _ = validate_subtitle_blocks([])
        assert is_valid

    def test_index_mismatch_fails(self) -> None:
        blocks = [
            SubtitleBlock(index=2, start_time=0.0, end_time=1.0, text="Out of order"),
        ]
        is_valid, err = validate_subtitle_blocks(blocks)
        assert not is_valid
        assert "index mismatch" in err

    def test_empty_text_fails(self) -> None:
        blocks = [
            SubtitleBlock(index=1, start_time=0.0, end_time=1.0, text="   "),
        ]
        is_valid, err = validate_subtitle_blocks(blocks)
        assert not is_valid
        assert "empty text" in err

    def test_valid_blocks_pass(self) -> None:
        blocks = [
            SubtitleBlock(index=1, start_time=0.0, end_time=2.0, text="Valid 1"),
            SubtitleBlock(index=2, start_time=2.1, end_time=4.0, text="Valid 2"),
        ]
        is_valid, _ = validate_subtitle_blocks(blocks)
        assert is_valid
