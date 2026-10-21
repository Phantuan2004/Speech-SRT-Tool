"""
Unit tests for Sentence Segmenter data models and interfaces.
"""

from speech_srt.asr_engine import WordTiming
from speech_srt.sentence_segmenter import SubtitleBlock, segment_into_subtitles


class TestSentenceSegmenterModels:
    """Tests for SubtitleBlock dataclass properties."""

    def test_subtitle_block_duration(self) -> None:
        block = SubtitleBlock(index=1, start_time=1.5, end_time=4.0, text="Sample text")
        assert block.duration == 2.5
        assert block.index == 1
        assert block.text == "Sample text"

    def test_word_timing_duration(self) -> None:
        word = WordTiming(word="hello", start=0.2, end=0.7, probability=0.98)
        assert abs(word.duration - 0.5) < 1e-6

    def test_segment_into_subtitles_empty_list(self) -> None:
        # Empty input words must gracefully return an empty list without error
        result = segment_into_subtitles([])
        assert result == []
