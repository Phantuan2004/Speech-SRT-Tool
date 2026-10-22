"""
Benchmark Evaluator Module.

Provides ground-truth loading, predicted-to-reference cue matching,
and computation of timestamp accuracy metrics for the Speech SRT Tool.

Metrics computed:
- Mean Absolute Error (MAE) for start and end timestamps.
- Percentage of timestamps within ±100ms, ±200ms, ±500ms.
- Word Error Rate (WER) when reference transcripts are available.
- Missing and spurious cue counts.
- Invalid cue count (zero/negative duration, non-chronological).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from speech_srt.sentence_segmenter import SubtitleBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ReferenceSentence:
    """A single ground-truth sentence with timing boundaries."""

    id: int
    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    difficulty: str = "normal"  # normal | noisy | fast | overlapping
    notes: str = ""


@dataclass
class BenchmarkEntry:
    """Ground-truth data for a single media file."""

    video_file: str
    language: str
    duration_seconds: float
    reference_sentences: List[ReferenceSentence]
    audio_file: Optional[str] = None
    notes: str = ""


@dataclass
class CueMatch:
    """Represents a matched pair of reference and predicted cues."""

    ref_id: int
    ref_text: str
    ref_start: float
    ref_end: float
    pred_index: int
    pred_text: str
    pred_start: float
    pred_end: float
    start_error: float  # absolute error in seconds
    end_error: float  # absolute error in seconds
    text_similarity: float  # 0.0 to 1.0

    @property
    def start_within_100ms(self) -> bool:
        return self.start_error <= 0.1

    @property
    def start_within_200ms(self) -> bool:
        return self.start_error <= 0.2

    @property
    def start_within_500ms(self) -> bool:
        return self.start_error <= 0.5

    @property
    def end_within_100ms(self) -> bool:
        return self.end_error <= 0.1

    @property
    def end_within_200ms(self) -> bool:
        return self.end_error <= 0.2

    @property
    def end_within_500ms(self) -> bool:
        return self.end_error <= 0.5


@dataclass
class BenchmarkMetrics:
    """Aggregated accuracy metrics for a benchmark evaluation run."""

    config_name: str
    video_file: str
    total_reference_cues: int = 0
    total_predicted_cues: int = 0
    matched_cues: int = 0
    missing_cues: int = 0  # Reference cues with no match
    spurious_cues: int = 0  # Predicted cues with no match
    invalid_cues: int = 0  # Predicted cues with bad timestamps

    # Timestamp error statistics (seconds)
    mean_start_error: float = 0.0
    mean_end_error: float = 0.0
    median_start_error: float = 0.0
    median_end_error: float = 0.0
    max_start_error: float = 0.0
    max_end_error: float = 0.0

    # Accuracy percentages
    pct_start_within_100ms: float = 0.0
    pct_start_within_200ms: float = 0.0
    pct_start_within_500ms: float = 0.0
    pct_end_within_100ms: float = 0.0
    pct_end_within_200ms: float = 0.0
    pct_end_within_500ms: float = 0.0
    pct_all_within_200ms: float = 0.0  # Both start AND end within ±200ms

    # Text accuracy
    mean_text_similarity: float = 0.0
    wer: Optional[float] = None

    # Processing info
    processing_time_seconds: Optional[float] = None

    matches: List[CueMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/CSV export."""
        result = {
            "config_name": self.config_name,
            "video_file": self.video_file,
            "total_reference_cues": self.total_reference_cues,
            "total_predicted_cues": self.total_predicted_cues,
            "matched_cues": self.matched_cues,
            "missing_cues": self.missing_cues,
            "spurious_cues": self.spurious_cues,
            "invalid_cues": self.invalid_cues,
            "mean_start_error_ms": round(self.mean_start_error * 1000, 1),
            "mean_end_error_ms": round(self.mean_end_error * 1000, 1),
            "median_start_error_ms": round(self.median_start_error * 1000, 1),
            "median_end_error_ms": round(self.median_end_error * 1000, 1),
            "max_start_error_ms": round(self.max_start_error * 1000, 1),
            "max_end_error_ms": round(self.max_end_error * 1000, 1),
            "pct_start_within_100ms": round(self.pct_start_within_100ms, 1),
            "pct_start_within_200ms": round(self.pct_start_within_200ms, 1),
            "pct_start_within_500ms": round(self.pct_start_within_500ms, 1),
            "pct_end_within_100ms": round(self.pct_end_within_100ms, 1),
            "pct_end_within_200ms": round(self.pct_end_within_200ms, 1),
            "pct_end_within_500ms": round(self.pct_end_within_500ms, 1),
            "pct_all_within_200ms": round(self.pct_all_within_200ms, 1),
            "mean_text_similarity": round(self.mean_text_similarity, 3),
            "wer": round(self.wer, 3) if self.wer is not None else None,
            "processing_time_seconds": (
                round(self.processing_time_seconds, 2)
                if self.processing_time_seconds is not None
                else None
            ),
        }
        return result


# ---------------------------------------------------------------------------
# Ground-Truth Loading
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> List[BenchmarkEntry]:
    """
    Load benchmark ground-truth entries from a JSON file.

    Args:
        path: Path to JSON file following the benchmark_format schema.

    Returns:
        List of BenchmarkEntry instances.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If JSON structure is invalid.
    """
    gt_path = Path(path).resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries_raw = data.get("entries", [])
    if not entries_raw:
        raise ValueError(f"Ground-truth file contains no entries: {gt_path}")

    entries: List[BenchmarkEntry] = []
    for entry_data in entries_raw:
        sentences: List[ReferenceSentence] = []
        for sent_data in entry_data.get("reference_sentences", []):
            sentences.append(
                ReferenceSentence(
                    id=sent_data["id"],
                    text=sent_data["text"],
                    start_time=float(sent_data["start_time"]),
                    end_time=float(sent_data["end_time"]),
                    difficulty=sent_data.get("difficulty", "normal"),
                    notes=sent_data.get("notes", ""),
                )
            )
        entries.append(
            BenchmarkEntry(
                video_file=entry_data["video_file"],
                audio_file=entry_data.get("audio_file"),
                language=entry_data.get("language", "en"),
                duration_seconds=float(entry_data["duration_seconds"]),
                reference_sentences=sentences,
                notes=entry_data.get("notes", ""),
            )
        )

    return entries


# ---------------------------------------------------------------------------
# Text Similarity (Token-level Jaccard for cue matching)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """
    Compute token-level Jaccard similarity between two text strings.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        Float between 0.0 (no overlap) and 1.0 (identical tokens).
    """
    tokens_a = set(_normalize_text(text_a).split())
    tokens_b = set(_normalize_text(text_b).split())

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) using edit distance.

    Args:
        reference: Ground-truth transcription text.
        hypothesis: ASR output text.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% error).
    """
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein distance on word tokens
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n


# ---------------------------------------------------------------------------
# Cue Matching (Greedy Temporal + Text Overlap)
# ---------------------------------------------------------------------------


def match_cues(
    reference: Sequence[ReferenceSentence],
    predicted: Sequence[SubtitleBlock],
    max_time_tolerance: float = 5.0,
    min_text_similarity: float = 0.2,
) -> Tuple[List[CueMatch], List[int], List[int]]:
    """
    Match predicted subtitle cues to reference sentences using temporal
    overlap and text similarity. Uses a greedy approach ordered by reference
    start time.

    Algorithm:
    1. For each reference cue (in chronological order), find all predicted
       cues whose time window overlaps within tolerance.
    2. Among candidates, pick the one with highest text similarity.
    3. Mark matched predicted cues as used (no double-matching).
    4. Unmatched reference → missing. Unmatched predicted → spurious.

    Args:
        reference: Ground-truth reference sentences.
        predicted: Model-generated subtitle cues.
        max_time_tolerance: Maximum temporal deviation in seconds to consider
            a predicted cue as a candidate match.
        min_text_similarity: Minimum Jaccard similarity threshold.

    Returns:
        Tuple of:
        - List of CueMatch objects for successful matches.
        - List of reference IDs that were not matched (missing cues).
        - List of predicted indices that were not matched (spurious cues).
    """
    if not reference or not predicted:
        missing = [r.id for r in reference] if reference else []
        spurious = [p.index for p in predicted] if predicted else []
        return [], missing, spurious

    matches: List[CueMatch] = []
    used_pred_indices: set = set()
    unmatched_ref_ids: List[int] = []

    for ref in reference:
        best_match: Optional[CueMatch] = None
        best_score: float = -1.0

        for pred in predicted:
            if pred.index in used_pred_indices:
                continue

            # Check temporal overlap: does the predicted cue overlap
            # with the reference window (expanded by tolerance)?
            ref_start_lo = ref.start_time - max_time_tolerance
            ref_end_hi = ref.end_time + max_time_tolerance

            # Candidate must overlap the reference time window
            if pred.end_time < ref_start_lo or pred.start_time > ref_end_hi:
                continue

            text_sim = compute_text_similarity(ref.text, pred.text)
            if text_sim < min_text_similarity:
                continue

            # Score: weighted combination of text similarity and temporal proximity
            time_center_ref = (ref.start_time + ref.end_time) / 2.0
            time_center_pred = (pred.start_time + pred.end_time) / 2.0
            temporal_proximity = 1.0 / (
                1.0 + abs(time_center_ref - time_center_pred)
            )

            score = 0.6 * text_sim + 0.4 * temporal_proximity

            if score > best_score:
                best_score = score
                best_match = CueMatch(
                    ref_id=ref.id,
                    ref_text=ref.text,
                    ref_start=ref.start_time,
                    ref_end=ref.end_time,
                    pred_index=pred.index,
                    pred_text=pred.text,
                    pred_start=pred.start_time,
                    pred_end=pred.end_time,
                    start_error=abs(pred.start_time - ref.start_time),
                    end_error=abs(pred.end_time - ref.end_time),
                    text_similarity=text_sim,
                )

        if best_match is not None:
            matches.append(best_match)
            used_pred_indices.add(best_match.pred_index)
        else:
            unmatched_ref_ids.append(ref.id)

    spurious_pred_indices = [
        p.index for p in predicted if p.index not in used_pred_indices
    ]

    return matches, unmatched_ref_ids, spurious_pred_indices


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def _count_invalid_cues(predicted: Sequence[SubtitleBlock]) -> int:
    """Count predicted cues with invalid timestamps."""
    count = 0
    prev_start = -1.0
    for cue in predicted:
        if cue.start_time < 0 or cue.end_time <= cue.start_time:
            count += 1
        elif cue.start_time < prev_start:
            count += 1
        prev_start = cue.start_time
    return count


def compute_metrics(
    config_name: str,
    video_file: str,
    reference: Sequence[ReferenceSentence],
    predicted: Sequence[SubtitleBlock],
    processing_time: Optional[float] = None,
) -> BenchmarkMetrics:
    """
    Compute comprehensive benchmark metrics from matched cues.

    Args:
        config_name: Identifier for the configuration being evaluated.
        video_file: Name of the source media file.
        reference: Ground-truth reference sentences.
        predicted: Model-generated subtitle cues.
        processing_time: Optional elapsed processing time in seconds.

    Returns:
        BenchmarkMetrics instance with all computed metrics.
    """
    matches, missing_ids, spurious_indices = match_cues(reference, predicted)

    metrics = BenchmarkMetrics(
        config_name=config_name,
        video_file=video_file,
        total_reference_cues=len(reference),
        total_predicted_cues=len(predicted),
        matched_cues=len(matches),
        missing_cues=len(missing_ids),
        spurious_cues=len(spurious_indices),
        invalid_cues=_count_invalid_cues(predicted),
        processing_time_seconds=processing_time,
        matches=matches,
    )

    if not matches:
        logger.warning(
            "No cues matched for '%s' / '%s'. Metrics will be zeroed.",
            config_name,
            video_file,
        )
        return metrics

    # Collect error arrays
    start_errors = [m.start_error for m in matches]
    end_errors = [m.end_error for m in matches]
    text_sims = [m.text_similarity for m in matches]

    n = len(matches)

    # Mean Absolute Errors
    metrics.mean_start_error = sum(start_errors) / n
    metrics.mean_end_error = sum(end_errors) / n

    # Median errors
    sorted_start = sorted(start_errors)
    sorted_end = sorted(end_errors)
    mid = n // 2
    if n % 2 == 1:
        metrics.median_start_error = sorted_start[mid]
        metrics.median_end_error = sorted_end[mid]
    else:
        metrics.median_start_error = (sorted_start[mid - 1] + sorted_start[mid]) / 2
        metrics.median_end_error = (sorted_end[mid - 1] + sorted_end[mid]) / 2

    # Max errors
    metrics.max_start_error = max(start_errors)
    metrics.max_end_error = max(end_errors)

    # Accuracy percentages (start timestamps)
    metrics.pct_start_within_100ms = (
        sum(1 for m in matches if m.start_within_100ms) / n * 100
    )
    metrics.pct_start_within_200ms = (
        sum(1 for m in matches if m.start_within_200ms) / n * 100
    )
    metrics.pct_start_within_500ms = (
        sum(1 for m in matches if m.start_within_500ms) / n * 100
    )

    # Accuracy percentages (end timestamps)
    metrics.pct_end_within_100ms = (
        sum(1 for m in matches if m.end_within_100ms) / n * 100
    )
    metrics.pct_end_within_200ms = (
        sum(1 for m in matches if m.end_within_200ms) / n * 100
    )
    metrics.pct_end_within_500ms = (
        sum(1 for m in matches if m.end_within_500ms) / n * 100
    )

    # Both start AND end within ±200ms
    metrics.pct_all_within_200ms = (
        sum(1 for m in matches if m.start_within_200ms and m.end_within_200ms)
        / n
        * 100
    )

    # Text similarity
    metrics.mean_text_similarity = sum(text_sims) / n

    # WER (concatenated reference vs concatenated prediction)
    ref_full_text = " ".join(r.text for r in reference)
    pred_full_text = " ".join(p.text for p in predicted)
    if ref_full_text.strip():
        metrics.wer = compute_wer(ref_full_text, pred_full_text)

    return metrics
