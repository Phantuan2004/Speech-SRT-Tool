"""
Tests for the Benchmark Evaluator Module.

Covers ground-truth loading, text similarity, WER computation,
cue matching algorithm, and metrics computation.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import List

from speech_srt.benchmark import (
    BenchmarkEntry,
    BenchmarkMetrics,
    CueMatch,
    ReferenceSentence,
    compute_metrics,
    compute_text_similarity,
    compute_wer,
    load_ground_truth,
    match_cues,
)
from speech_srt.sentence_segmenter import SubtitleBlock


class TestTextSimilarity(unittest.TestCase):
    """Tests for token-level Jaccard similarity."""

    def test_identical_texts(self) -> None:
        self.assertAlmostEqual(
            compute_text_similarity("Hello world", "Hello world"), 1.0
        )

    def test_completely_different(self) -> None:
        self.assertAlmostEqual(
            compute_text_similarity("Hello world", "Goodbye moon"), 0.0
        )

    def test_partial_overlap(self) -> None:
        sim = compute_text_similarity("The quick brown fox", "The slow brown dog")
        # Shared: "the", "brown" → 2/6
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_case_insensitive(self) -> None:
        self.assertAlmostEqual(
            compute_text_similarity("Hello World", "hello world"), 1.0
        )

    def test_punctuation_ignored(self) -> None:
        self.assertAlmostEqual(
            compute_text_similarity("Hello, world!", "Hello world"), 1.0
        )

    def test_both_empty(self) -> None:
        self.assertAlmostEqual(compute_text_similarity("", ""), 1.0)

    def test_one_empty(self) -> None:
        self.assertAlmostEqual(compute_text_similarity("Hello", ""), 0.0)


class TestWordErrorRate(unittest.TestCase):
    """Tests for WER computation."""

    def test_perfect_match(self) -> None:
        wer = compute_wer("the cat sat on the mat", "the cat sat on the mat")
        self.assertAlmostEqual(wer, 0.0)

    def test_complete_mismatch(self) -> None:
        wer = compute_wer("hello world", "goodbye moon")
        self.assertAlmostEqual(wer, 1.0)

    def test_single_insertion(self) -> None:
        wer = compute_wer("the cat sat", "the big cat sat")
        # 1 insertion / 3 ref words
        self.assertAlmostEqual(wer, 1 / 3, places=2)

    def test_single_deletion(self) -> None:
        wer = compute_wer("the big cat sat", "the cat sat")
        # 1 deletion / 4 ref words
        self.assertAlmostEqual(wer, 1 / 4, places=2)

    def test_empty_reference(self) -> None:
        self.assertAlmostEqual(compute_wer("", ""), 0.0)

    def test_empty_ref_nonempty_hyp(self) -> None:
        self.assertAlmostEqual(compute_wer("", "hello world"), 1.0)


class TestCueMatching(unittest.TestCase):
    """Tests for reference-to-predicted cue matching."""

    def _make_ref(
        self, sentences: List[tuple],
    ) -> List[ReferenceSentence]:
        return [
            ReferenceSentence(id=i + 1, text=text, start_time=s, end_time=e)
            for i, (text, s, e) in enumerate(sentences)
        ]

    def _make_pred(
        self, cues: List[tuple],
    ) -> List[SubtitleBlock]:
        return [
            SubtitleBlock(index=i + 1, start_time=s, end_time=e, text=text)
            for i, (text, s, e) in enumerate(cues)
        ]

    def test_perfect_alignment(self) -> None:
        ref = self._make_ref([
            ("Hello world", 1.0, 3.0),
            ("Goodbye moon", 4.0, 6.0),
        ])
        pred = self._make_pred([
            ("Hello world", 1.0, 3.0),
            ("Goodbye moon", 4.0, 6.0),
        ])

        matches, missing, spurious = match_cues(ref, pred)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(spurious), 0)
        self.assertAlmostEqual(matches[0].start_error, 0.0)
        self.assertAlmostEqual(matches[0].end_error, 0.0)

    def test_slight_offset(self) -> None:
        ref = self._make_ref([("Hello world", 1.0, 3.0)])
        pred = self._make_pred([("Hello world", 1.15, 3.10)])

        matches, missing, spurious = match_cues(ref, pred)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].start_error, 0.15)
        self.assertAlmostEqual(matches[0].end_error, 0.10)

    def test_missing_cue(self) -> None:
        ref = self._make_ref([
            ("Hello world", 1.0, 3.0),
            ("Goodbye moon", 4.0, 6.0),
        ])
        pred = self._make_pred([("Hello world", 1.0, 3.0)])

        matches, missing, spurious = match_cues(ref, pred)
        self.assertEqual(len(matches), 1)
        self.assertEqual(missing, [2])
        self.assertEqual(len(spurious), 0)

    def test_spurious_cue(self) -> None:
        ref = self._make_ref([("Hello world", 1.0, 3.0)])
        pred = self._make_pred([
            ("Hello world", 1.0, 3.0),
            ("Extra subtitle", 5.0, 7.0),
        ])

        matches, missing, spurious = match_cues(ref, pred)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(missing), 0)
        self.assertEqual(spurious, [2])

    def test_empty_reference(self) -> None:
        pred = self._make_pred([("Hello", 1.0, 3.0)])
        matches, missing, spurious = match_cues([], pred)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(missing), 0)
        self.assertEqual(spurious, [1])

    def test_empty_predicted(self) -> None:
        ref = self._make_ref([("Hello", 1.0, 3.0)])
        matches, missing, spurious = match_cues(ref, [])
        self.assertEqual(len(matches), 0)
        self.assertEqual(missing, [1])
        self.assertEqual(len(spurious), 0)

    def test_text_mismatch_prevents_match(self) -> None:
        """Completely different text should not match even with same timing."""
        ref = self._make_ref([("Alpha beta gamma", 1.0, 3.0)])
        pred = self._make_pred([("Delta epsilon zeta", 1.0, 3.0)])

        matches, missing, spurious = match_cues(ref, pred)
        self.assertEqual(len(matches), 0)
        self.assertEqual(missing, [1])
        self.assertEqual(spurious, [1])


class TestMetricsComputation(unittest.TestCase):
    """Tests for the full metrics computation pipeline."""

    def test_perfect_score(self) -> None:
        ref = [
            ReferenceSentence(id=1, text="Hello world", start_time=1.0, end_time=3.0),
            ReferenceSentence(id=2, text="Second sentence", start_time=4.0, end_time=6.0),
        ]
        pred = [
            SubtitleBlock(index=1, start_time=1.0, end_time=3.0, text="Hello world"),
            SubtitleBlock(index=2, start_time=4.0, end_time=6.0, text="Second sentence"),
        ]

        metrics = compute_metrics("test", "video.mp4", ref, pred)
        self.assertEqual(metrics.matched_cues, 2)
        self.assertEqual(metrics.missing_cues, 0)
        self.assertEqual(metrics.spurious_cues, 0)
        self.assertAlmostEqual(metrics.mean_start_error, 0.0)
        self.assertAlmostEqual(metrics.mean_end_error, 0.0)
        self.assertAlmostEqual(metrics.pct_all_within_200ms, 100.0)

    def test_moderate_accuracy(self) -> None:
        ref = [
            ReferenceSentence(id=1, text="Hello world", start_time=1.0, end_time=3.0),
            ReferenceSentence(id=2, text="This is a test", start_time=4.0, end_time=6.0),
        ]
        pred = [
            SubtitleBlock(index=1, start_time=1.15, end_time=3.10, text="Hello world"),
            SubtitleBlock(index=2, start_time=4.50, end_time=6.80, text="This is a test"),
        ]

        metrics = compute_metrics("test", "video.mp4", ref, pred)
        self.assertEqual(metrics.matched_cues, 2)
        # First match: within 200ms. Second: 500ms start, 800ms end → not within 200ms.
        self.assertAlmostEqual(metrics.pct_all_within_200ms, 50.0)

    def test_to_dict_serialization(self) -> None:
        ref = [
            ReferenceSentence(id=1, text="Hello", start_time=1.0, end_time=3.0),
        ]
        pred = [
            SubtitleBlock(index=1, start_time=1.0, end_time=3.0, text="Hello"),
        ]

        metrics = compute_metrics("config_x", "test.mp4", ref, pred)
        d = metrics.to_dict()
        self.assertEqual(d["config_name"], "config_x")
        self.assertEqual(d["video_file"], "test.mp4")
        self.assertEqual(d["matched_cues"], 1)
        self.assertIn("mean_start_error_ms", d)
        self.assertIn("pct_all_within_200ms", d)


class TestGroundTruthLoading(unittest.TestCase):
    """Tests for loading ground-truth JSON files."""

    def test_load_valid_format(self) -> None:
        """Loading the schema template should parse without errors."""
        fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "benchmark_format.json"
        )
        entries = load_ground_truth(fixture_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].video_file, "sample_video.mp4")
        self.assertEqual(len(entries[0].reference_sentences), 2)
        self.assertEqual(entries[0].reference_sentences[0].text, "This is an example reference sentence.")
        self.assertAlmostEqual(entries[0].reference_sentences[0].start_time, 1.200)

    def test_load_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ground_truth(Path("nonexistent_file.json"))


if __name__ == "__main__":
    unittest.main()
