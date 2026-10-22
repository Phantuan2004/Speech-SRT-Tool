"""
Word Aligner Module.

Refines word-level timestamp boundaries to improve subtitle timing accuracy.
Implements monotonic enforcement, overlap resolution, and gap interpolation
to achieve target accuracy within ±200 ms on at least 95% of benchmarked timestamps.

This module works entirely on WordTiming objects from the ASR engine output
without requiring external alignment models or network access.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from speech_srt.asr_engine import WordTiming, TranscriptionSegment

logger = logging.getLogger(__name__)


def refine_word_timings(
    words: List[WordTiming],
    min_word_duration: float = 0.05,
    max_gap_interpolation: float = 0.3,
) -> List[WordTiming]:
    """
    Refine and sanitize raw word-level timestamps from ASR output.

    Processing steps:
    1. Remove words with empty text.
    2. Enforce strictly non-decreasing monotonic start timestamps.
    3. Ensure minimum word duration.
    4. Resolve overlaps between neighboring words.
    5. Interpolate small gaps between consecutive words for smoother timing.

    Args:
        words: List of WordTiming objects from ASR engine.
        min_word_duration: Minimum allowed duration per word in seconds.
        max_gap_interpolation: Maximum gap size (seconds) to interpolate.
            Gaps larger than this are preserved as natural pauses.

    Returns:
        List of cleaned, monotonic WordTiming objects.
    """
    if not words:
        return []

    # Step 1: Filter out empty words
    cleaned = [w for w in words if w.word.strip()]
    if not cleaned:
        return []

    # Step 2: Create mutable copies to avoid mutating input
    refined: List[WordTiming] = [
        WordTiming(
            word=w.word,
            start=w.start,
            end=w.end,
            probability=w.probability,
        )
        for w in cleaned
    ]

    # Step 3: Enforce monotonic start times
    _enforce_monotonic_starts(refined)

    # Step 4: Enforce minimum word duration
    _enforce_min_duration(refined, min_word_duration)

    # Step 5: Resolve overlaps between consecutive words
    _resolve_overlaps(refined)

    # Step 6: Interpolate small gaps
    _interpolate_gaps(refined, max_gap_interpolation)

    return refined


def _enforce_monotonic_starts(words: List[WordTiming]) -> None:
    """
    Ensure word start times are strictly non-decreasing.
    If a word starts before the previous word, push its start to
    the previous word's end time.

    Args:
        words: List of WordTiming (modified in-place).
    """
    for i in range(1, len(words)):
        if words[i].start < words[i - 1].start:
            logger.debug(
                "Monotonic fix: word '%s' start %.3fs pushed to %.3fs",
                words[i].word,
                words[i].start,
                words[i - 1].end,
            )
            words[i].start = words[i - 1].end

        # Also ensure start < end for current word
        if words[i].start >= words[i].end:
            words[i].end = words[i].start + 0.05


def _enforce_min_duration(
    words: List[WordTiming], min_duration: float
) -> None:
    """
    Ensure every word has at least `min_duration` seconds.

    Args:
        words: List of WordTiming (modified in-place).
        min_duration: Minimum allowed duration per word.
    """
    for w in words:
        actual_duration = w.end - w.start
        if actual_duration < min_duration:
            w.end = w.start + min_duration


def _resolve_overlaps(words: List[WordTiming]) -> None:
    """
    Resolve temporal overlaps between consecutive words by moving
    boundaries to the midpoint of the overlapping region.

    Args:
        words: List of WordTiming (modified in-place).
    """
    for i in range(1, len(words)):
        if words[i].start < words[i - 1].end:
            overlap = words[i - 1].end - words[i].start
            midpoint = words[i].start + overlap / 2.0
            logger.debug(
                "Overlap fix: '%s' end %.3fs and '%s' start %.3fs → midpoint %.3fs",
                words[i - 1].word,
                words[i - 1].end,
                words[i].word,
                words[i].start,
                midpoint,
            )
            words[i - 1].end = midpoint
            words[i].start = midpoint

            # Enforce end > start after midpoint adjustment
            if words[i - 1].end <= words[i - 1].start:
                words[i - 1].end = words[i - 1].start + 0.02
            if words[i].end <= words[i].start:
                words[i].end = words[i].start + 0.02


def _interpolate_gaps(
    words: List[WordTiming], max_gap: float
) -> None:
    """
    For small gaps between consecutive words, distribute the gap
    evenly by extending the previous word's end and the next word's start.
    This reduces perceived timing jumps in subtitle display.

    Gaps larger than max_gap are treated as intentional pauses and preserved.

    Args:
        words: List of WordTiming (modified in-place).
        max_gap: Maximum gap size to interpolate.
    """
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if 0.0 < gap <= max_gap:
            half_gap = gap / 2.0
            words[i - 1].end += half_gap
            words[i].start -= half_gap


def refine_segments(
    segments: List[TranscriptionSegment],
    min_word_duration: float = 0.05,
    max_gap_interpolation: float = 0.3,
) -> List[TranscriptionSegment]:
    """
    Apply word-level timestamp refinement to all segments.

    First collects all words across segments, refines them globally
    (ensuring cross-segment monotonicity), then redistributes them
    back to their original segments.

    Args:
        segments: List of TranscriptionSegment from ASR.
        min_word_duration: Minimum duration per word.
        max_gap_interpolation: Max gap to interpolate.

    Returns:
        New list of TranscriptionSegment with refined word timings.
    """
    if not segments:
        return []

    # Collect all words with segment ownership tracking
    all_words: List[WordTiming] = []
    segment_boundaries: List[int] = []  # indices into all_words

    for seg in segments:
        segment_boundaries.append(len(all_words))
        if seg.words:
            all_words.extend(seg.words)

    # Global refinement across all words
    refined_words = refine_word_timings(
        all_words,
        min_word_duration=min_word_duration,
        max_gap_interpolation=max_gap_interpolation,
    )

    if not refined_words:
        return segments

    # Redistribute refined words back to segments
    # Map refined words back using text matching and order
    refined_segments: List[TranscriptionSegment] = []
    word_idx = 0

    for seg_idx, seg in enumerate(segments):
        if not seg.words:
            refined_segments.append(seg)
            continue

        # Take the next len(seg.words) refined words for this segment
        num_words = len(seg.words)
        seg_refined = refined_words[word_idx : word_idx + num_words]
        word_idx += num_words

        if seg_refined:
            new_start = seg_refined[0].start
            new_end = seg_refined[-1].end
        else:
            new_start = seg.start
            new_end = seg.end

        refined_segments.append(
            TranscriptionSegment(
                id=seg.id,
                seek=seg.seek,
                start=new_start,
                end=new_end,
                text=seg.text,
                words=seg_refined,
            )
        )

    return refined_segments
