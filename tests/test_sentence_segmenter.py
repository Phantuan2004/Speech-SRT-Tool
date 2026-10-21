"""
Unit tests for Sentence Segmenter Module.
"""

from speech_srt.asr_engine import TranscriptionSegment, WordTiming
from speech_srt.sentence_segmenter import (
    SubtitleBlock,
    segment_into_subtitles,
    segment_transcription_result,
    segment_words_into_cues,
    wrap_cue_text,
)


class TestSentenceSegmenter:
    """Tests for sentence cue segmentation logic."""

    def test_empty_words_returns_empty_list(self) -> None:
        assert segment_words_into_cues([]) == []
        assert segment_into_subtitles([]) == []
        assert segment_transcription_result([]) == []

    def test_split_on_sentence_punctuation(self) -> None:
        words = [
            WordTiming(word="Hello", start=0.0, end=0.4),
            WordTiming(word="world.", start=0.5, end=0.9),  # Period ends sentence
            WordTiming(word="How", start=1.0, end=1.3),
            WordTiming(word="are", start=1.4, end=1.6),
            WordTiming(word="you?", start=1.7, end=2.0),
        ]
        cues = segment_words_into_cues(words, max_pause_seconds=2.0)
        assert len(cues) == 2
        assert cues[0].text == "Hello world."
        assert cues[0].start_time == 0.0
        assert cues[0].end_time == 0.9

        assert cues[1].text == "How are you?"
        assert cues[1].start_time == 1.0
        assert cues[1].end_time == 2.0

    def test_split_on_acoustic_pause_gap(self) -> None:
        words = [
            WordTiming(word="First", start=0.0, end=0.5),
            WordTiming(word="part", start=0.6, end=1.0),
            # 1.2s silence gap here
            WordTiming(word="Second", start=2.2, end=2.6),
            WordTiming(word="part", start=2.7, end=3.1),
        ]
        cues = segment_words_into_cues(words, max_pause_seconds=0.8)
        assert len(cues) == 2
        assert cues[0].text == "First part"
        assert cues[0].start_time == 0.0
        assert cues[0].end_time == 1.0

        assert cues[1].text == "Second part"
        assert cues[1].start_time == 2.2
        assert cues[1].end_time == 3.1

    def test_wrap_cue_text_under_limit(self) -> None:
        tokens = ["This", "is", "a", "short", "line."]
        wrapped = wrap_cue_text(tokens, max_line_length=42, max_lines=2)
        assert "\n" not in wrapped
        assert wrapped == "This is a short line."

    def test_wrap_cue_text_over_limit_splits_into_two_lines(self) -> None:
        tokens = [
            "This", "is", "a", "rather", "long", "sentence",
            "that", "needs", "to", "be", "wrapped", "cleanly",
            "for", "CapCut", "display", "standard.",
        ]
        wrapped = wrap_cue_text(tokens, max_line_length=35, max_lines=2)
        lines = wrapped.splitlines()
        assert len(lines) == 2
        assert len(lines[0]) <= 35
        # Ensure words weren't dropped
        assert "CapCut" in lines[1]

    def test_fallback_when_segment_lacks_words(self) -> None:
        seg = TranscriptionSegment(
            id=1,
            seek=0,
            start=2.0,
            end=5.5,
            text="Sentence without word-level timestamps",
            words=[],  # No word-level timings
        )
        cues = segment_transcription_result([seg])
        assert len(cues) == 1
        assert cues[0].text == "Sentence without word-level timestamps"
        assert cues[0].start_time == 2.0
        assert cues[0].end_time == 5.5

    def test_cues_have_positive_non_zero_duration(self) -> None:
        words = [
            WordTiming(word="Click", start=1.0, end=1.0),  # degenerate timestamp
        ]
        cues = segment_words_into_cues(words)
        assert len(cues) == 1
        assert cues[0].end_time > cues[0].start_time
