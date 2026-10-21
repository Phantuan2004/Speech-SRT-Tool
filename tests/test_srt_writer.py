"""
Unit tests for SRT Serializer, Writer, and Parser.
"""

import pytest
from pathlib import Path
from speech_srt.sentence_segmenter import SubtitleBlock
from speech_srt.srt_writer import (
    SRTFormatError,
    format_timestamp,
    parse_srt,
    parse_timestamp,
    serialize_srt,
    write_srt,
)


class TestFormatTimestamp:
    """Tests for format_timestamp function ensuring CapCut timestamp compliance."""

    def test_zero_timestamp(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_sub_second_precision(self) -> None:
        assert format_timestamp(0.125) == "00:00:00,125"
        assert format_timestamp(0.005) == "00:00:00,005"

    def test_seconds_and_minutes(self) -> None:
        assert format_timestamp(65.432) == "00:01:05,432"

    def test_hour_rollover(self) -> None:
        assert format_timestamp(3661.005) == "01:01:01,005"

    def test_negative_timestamp_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            format_timestamp(-1.0)


class TestParseTimestamp:
    """Tests for parse_timestamp function."""

    def test_parse_valid_timestamp(self) -> None:
        assert abs(parse_timestamp("00:01:05,432") - 65.432) < 1e-4
        assert abs(parse_timestamp("01:01:01,005") - 3661.005) < 1e-4

    def test_parse_invalid_timestamp_raises_error(self) -> None:
        with pytest.raises(SRTFormatError, match="Invalid SRT timestamp"):
            parse_timestamp("00:01:05.432")  # Dot instead of comma


class TestSerializeAndParseRoundtrip:
    """Tests for serialize_srt and parse_srt roundtrip fidelity."""

    def test_empty_blocks(self) -> None:
        assert serialize_srt([]) == ""
        assert parse_srt("") == []

    def test_roundtrip_multi_blocks(self, tmp_path: Path) -> None:
        blocks = [
            SubtitleBlock(index=1, start_time=1.250, end_time=3.500, text="First line of dialogue."),
            SubtitleBlock(index=2, start_time=3.800, end_time=6.120, text="Second line.\nWith two rows."),
        ]

        srt_file = tmp_path / "roundtrip.srt"
        write_srt(blocks, srt_file)

        # Re-parse from file
        parsed = parse_srt(srt_file)
        assert len(parsed) == 2

        assert parsed[0].index == 1
        assert abs(parsed[0].start_time - 1.250) < 1e-3
        assert abs(parsed[0].end_time - 3.500) < 1e-3
        assert parsed[0].text == "First line of dialogue."

        assert parsed[1].index == 2
        assert abs(parsed[1].start_time - 3.800) < 1e-3
        assert abs(parsed[1].end_time - 6.120) < 1e-3
        assert parsed[1].text == "Second line.\nWith two rows."

    def test_parse_malformed_srt_raises_error(self) -> None:
        malformed = "1\nInvalid Arrow Timestamp Line\nHello"
        with pytest.raises(SRTFormatError):
            parse_srt(malformed)
