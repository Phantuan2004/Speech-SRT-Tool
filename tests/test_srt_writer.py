"""
Unit tests for SRT Serializer and Formatter.
"""

import pytest
from pathlib import Path
from speech_srt.sentence_segmenter import SubtitleBlock
from speech_srt.srt_writer import format_timestamp, serialize_srt, write_srt


class TestFormatTimestamp:
    """Tests for format_timestamp function ensuring CapCut timestamp compliance."""

    def test_zero_timestamp(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_sub_second_precision(self) -> None:
        assert format_timestamp(0.125) == "00:00:00,125"
        assert format_timestamp(0.005) == "00:00:00,005"

    def test_seconds_and_minutes(self) -> None:
        # 1 minute, 5 seconds, 432 milliseconds
        assert format_timestamp(65.432) == "00:01:05,432"

    def test_hour_rollover(self) -> None:
        # 1 hour, 1 minute, 1 second, 5 milliseconds
        assert format_timestamp(3661.005) == "01:01:01,005"

    def test_multi_hour(self) -> None:
        assert format_timestamp(7325.890) == "02:02:05,890"

    def test_negative_timestamp_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            format_timestamp(-1.0)


class TestSerializeSrt:
    """Tests for serialize_srt function."""

    def test_empty_blocks_returns_empty_string(self) -> None:
        assert serialize_srt([]) == ""

    def test_single_block_formatting(self) -> None:
        blocks = [
            SubtitleBlock(index=1, start_time=1.0, end_time=3.5, text="Hello world!"),
        ]
        expected = "1\n00:00:01,000 --> 00:00:03,500\nHello world!\n"
        assert serialize_srt(blocks) == expected

    def test_multiple_blocks_formatting(self) -> None:
        blocks = [
            SubtitleBlock(index=1, start_time=0.5, end_time=2.0, text="First subtitle."),
            SubtitleBlock(index=2, start_time=2.2, end_time=4.1, text="Second subtitle."),
        ]
        result = serialize_srt(blocks)
        assert "1\n00:00:00,500 --> 00:00:02,000\nFirst subtitle." in result
        assert "2\n00:00:02,200 --> 00:00:04,100\nSecond subtitle." in result


class TestWriteSrt:
    """Tests for file writing functionality."""

    def test_write_srt_creates_valid_utf8_file(self, tmp_path: Path) -> None:
        out_file = tmp_path / "test_output.srt"
        blocks = [
            SubtitleBlock(index=1, start_time=0.0, end_time=2.5, text="Testing CapCut export."),
        ]
        result_path = write_srt(blocks, out_file)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "Testing CapCut export." in content
