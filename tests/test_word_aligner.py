"""
Tests for the Word Aligner Module.

Covers monotonic enforcement, minimum duration, overlap resolution,
gap interpolation, and cross-segment refinement.
"""

from __future__ import annotations

import unittest
from typing import List

from speech_srt.asr_engine import TranscriptionSegment, WordTiming
from speech_srt.word_aligner import (
    refine_segments,
    refine_word_timings,
)


class TestRefineWordTimings(unittest.TestCase):
    """Tests for the refine_word_timings function."""

    def _make_words(self, data: List[tuple]) -> List[WordTiming]:
        """Helper: create WordTiming list from (word, start, end) tuples."""
        return [
            WordTiming(word=w, start=s, end=e, probability=0.9)
            for w, s, e in data
        ]

    def test_empty_input(self) -> None:
        result = refine_word_timings([])
        self.assertEqual(result, [])

    def test_single_word_passthrough(self) -> None:
        words = self._make_words([("Hello", 1.0, 1.5)])
        result = refine_word_timings(words)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 1.0)
        self.assertAlmostEqual(result[0].end, 1.5)

    def test_monotonic_order_preserved(self) -> None:
        words = self._make_words([
            ("Hello", 1.0, 1.5),
            ("world", 2.0, 2.5),
            ("test", 3.0, 3.5),
        ])
        result = refine_word_timings(words)
        for i in range(1, len(result)):
            self.assertGreaterEqual(
                result[i].start,
                result[i - 1].start,
                f"Word {i} starts before word {i-1}",
            )

    def test_non_monotonic_starts_fixed(self) -> None:
        """Words with decreasing start times should be pushed forward."""
        words = self._make_words([
            ("First", 2.0, 2.5),
            ("Second", 1.5, 2.0),  # starts before First
            ("Third", 3.0, 3.5),
        ])
        result = refine_word_timings(words)
        self.assertGreaterEqual(result[1].start, result[0].end)
        self.assertLess(result[1].start, result[1].end)

    def test_min_duration_enforced(self) -> None:
        """Very short words should be extended to minimum duration."""
        words = self._make_words([
            ("a", 1.0, 1.01),  # Only 10ms
            ("be", 2.0, 2.5),
        ])
        result = refine_word_timings(words, min_word_duration=0.05)
        self.assertGreaterEqual(result[0].end - result[0].start, 0.05)

    def test_overlap_resolved(self) -> None:
        """Overlapping words should be separated at midpoint."""
        words = self._make_words([
            ("Hello", 1.0, 2.0),
            ("world", 1.5, 3.0),  # Overlaps with Hello
        ])
        result = refine_word_timings(words)
        self.assertGreaterEqual(
            result[1].start,
            result[0].end,
            "Overlap should be resolved",
        )

    def test_small_gaps_interpolated(self) -> None:
        """Small gaps between words should be closed by interpolation."""
        words = self._make_words([
            ("Hello", 1.0, 1.5),
            ("world", 1.7, 2.2),  # 200ms gap
        ])
        result = refine_word_timings(words, max_gap_interpolation=0.3)
        gap = result[1].start - result[0].end
        self.assertLess(gap, 0.2, "Gap should be reduced by interpolation")

    def test_large_gaps_preserved(self) -> None:
        """Large gaps (natural pauses) should not be interpolated."""
        words = self._make_words([
            ("Hello", 1.0, 1.5),
            ("world", 3.0, 3.5),  # 1.5s gap
        ])
        result = refine_word_timings(words, max_gap_interpolation=0.3)
        gap = result[1].start - result[0].end
        self.assertAlmostEqual(gap, 1.5, places=1)

    def test_empty_words_filtered(self) -> None:
        words = [
            WordTiming(word="Hello", start=1.0, end=1.5),
            WordTiming(word="   ", start=2.0, end=2.5),
            WordTiming(word="world", start=3.0, end=3.5),
        ]
        result = refine_word_timings(words)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].word, "Hello")
        self.assertEqual(result[1].word, "world")

    def test_all_words_have_positive_duration(self) -> None:
        """Every refined word must have end > start."""
        words = self._make_words([
            ("a", 1.0, 1.0),    # Zero duration
            ("b", 1.0, 0.5),    # Negative duration
            ("c", 2.0, 2.5),    # Normal
        ])
        result = refine_word_timings(words)
        for w in result:
            self.assertGreater(
                w.end, w.start,
                f"Word '{w.word}' has invalid duration: {w.start} -> {w.end}",
            )

    def test_input_not_mutated(self) -> None:
        """Original input should not be modified."""
        words = self._make_words([
            ("Hello", 2.0, 2.5),
            ("world", 1.5, 2.0),
        ])
        original_starts = [w.start for w in words]
        _ = refine_word_timings(words)
        for w, orig_start in zip(words, original_starts):
            self.assertAlmostEqual(w.start, orig_start)


class TestRefineSegments(unittest.TestCase):
    """Tests for cross-segment refinement."""

    def test_empty_segments(self) -> None:
        result = refine_segments([])
        self.assertEqual(result, [])

    def test_segments_without_words(self) -> None:
        segments = [
            TranscriptionSegment(id=0, seek=0, start=1.0, end=3.0, text="Hello", words=[]),
        ]
        result = refine_segments(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hello")

    def test_segment_timing_updated_from_words(self) -> None:
        words = [
            WordTiming(word="Hello", start=1.0, end=1.5),
            WordTiming(word="world", start=2.0, end=2.5),
        ]
        segments = [
            TranscriptionSegment(
                id=0, seek=0, start=0.5, end=3.0, text="Hello world", words=words
            ),
        ]
        result = refine_segments(segments)
        self.assertEqual(len(result), 1)
        # Segment timing should match refined word boundaries
        self.assertAlmostEqual(result[0].start, result[0].words[0].start)
        self.assertAlmostEqual(result[0].end, result[0].words[-1].end)

    def test_cross_segment_monotonicity(self) -> None:
        """Words across different segments should maintain monotonic order."""
        seg1_words = [
            WordTiming(word="Hello", start=1.0, end=1.5),
            WordTiming(word="world", start=2.0, end=2.5),
        ]
        seg2_words = [
            WordTiming(word="Good", start=3.0, end=3.5),
            WordTiming(word="morning", start=4.0, end=4.5),
        ]
        segments = [
            TranscriptionSegment(
                id=0, seek=0, start=1.0, end=2.5, text="Hello world", words=seg1_words
            ),
            TranscriptionSegment(
                id=1, seek=0, start=3.0, end=4.5, text="Good morning", words=seg2_words
            ),
        ]
        result = refine_segments(segments)
        self.assertEqual(len(result), 2)

        # Last word of seg1 should end before or at first word of seg2
        last_word_seg1 = result[0].words[-1]
        first_word_seg2 = result[1].words[0]
        self.assertLessEqual(last_word_seg1.end, first_word_seg2.start + 0.01)


if __name__ == "__main__":
    unittest.main()
