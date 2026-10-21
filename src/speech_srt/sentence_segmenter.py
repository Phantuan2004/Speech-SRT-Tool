"""
Sentence Segmentation Module for Subtitles.

Groups word-level timings into coherent, visually balanced subtitle cues adhering to
CapCut display constraints (e.g. max 42 characters per line, max 2 lines per subtitle).
Preserves natural speech pauses, sentence punctuation, and exact acoustic timing boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence
from speech_srt.asr_engine import WordTiming, TranscriptionSegment

logger = logging.getLogger(__name__)

SENTENCE_END_PUNCTUATION = {".", "!", "?", "...", "…"}
CLAUSE_PUNCTUATION = {",", ";", ":", "-", "—"}


@dataclass
class SubtitleBlock:
    """Represents a single formatted subtitle entry in an SRT file."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def wrap_cue_text(words_text: Sequence[str], max_line_length: int = 42, max_lines: int = 2) -> str:
    """
    Wrap words into at most max_lines lines without breaking words or dropping tokens.

    Args:
        words_text: List of word tokens.
        max_line_length: Max characters per line.
        max_lines: Max lines allowed in a subtitle block (default: 2).

    Returns:
        Formatted multi-line subtitle string.
    """
    tokens = [w.strip() for w in words_text if w.strip()]
    if not tokens:
        return ""

    full_text = " ".join(tokens)
    if len(full_text) <= max_line_length or max_lines <= 1:
        return full_text

    lines: List[str] = []
    idx = 0
    while idx < len(tokens):
        if len(lines) == max_lines - 1:
            # Last allowed line: collect all remaining tokens
            lines.append(" ".join(tokens[idx:]))
            break

        current_line: List[str] = []
        while idx < len(tokens):
            candidate = " ".join(current_line + [tokens[idx]])
            if len(candidate) <= max_line_length or not current_line:
                current_line.append(tokens[idx])
                idx += 1
            else:
                break

        lines.append(" ".join(current_line))

    return "\n".join(lines)


def segment_words_into_cues(
    words: Sequence[WordTiming],
    start_index: int = 1,
    max_line_length: int = 42,
    max_lines_per_subtitle: int = 2,
    max_pause_seconds: float = 0.6,
    max_cue_duration: float = 6.0,
) -> List[SubtitleBlock]:
    """
    Group words into subtitle cues based on punctuation, pause gaps, and length limits.

    Args:
        words: Aligned WordTiming objects.
        start_index: 1-based index offset for sequential numbering.
        max_line_length: Maximum characters per line.
        max_lines_per_subtitle: Maximum lines per subtitle cue (default: 2).
        max_pause_seconds: Silence gap between words triggering a cue break.
        max_cue_duration: Maximum allowed time duration for a single subtitle cue.

    Returns:
        List of SubtitleBlock instances.
    """
    if not words:
        return []

    cues: List[SubtitleBlock] = []
    current_chunk: List[WordTiming] = []
    max_total_chars = max_line_length * max_lines_per_subtitle

    for word in words:
        if not current_chunk:
            current_chunk.append(word)
            continue

        prev_word = current_chunk[-1]
        pause_gap = word.start - prev_word.end
        cue_duration = word.end - current_chunk[0].start
        current_text = " ".join(w.word.strip() for w in current_chunk)
        new_text_len = len(current_text) + 1 + len(word.word.strip())

        # Determine if we should split
        should_split = False

        # 1. Natural acoustic silence gap
        if pause_gap >= max_pause_seconds:
            should_split = True
        # 2. Punctuation ending a sentence
        elif any(prev_word.word.rstrip().endswith(punct) for punct in SENTENCE_END_PUNCTUATION):
            should_split = True
        # 3. Maximum duration exceeded
        elif cue_duration > max_cue_duration:
            should_split = True
        # 4. Total characters exceed limits
        elif new_text_len > max_total_chars:
            should_split = True
        # 5. Clause punctuation break when already reasonably full
        elif any(prev_word.word.rstrip().endswith(punct) for punct in CLAUSE_PUNCTUATION) and len(current_text) >= max_line_length:
            should_split = True

        if should_split and current_chunk:
            cue_start = current_chunk[0].start
            cue_end = current_chunk[-1].end
            # Enforce non-zero positive duration
            if cue_end <= cue_start:
                cue_end = cue_start + 0.1

            cue_text = wrap_cue_text(
                [w.word for w in current_chunk],
                max_line_length=max_line_length,
                max_lines=max_lines_per_subtitle,
            )

            cues.append(
                SubtitleBlock(
                    index=start_index + len(cues),
                    start_time=cue_start,
                    end_time=cue_end,
                    text=cue_text,
                )
            )
            current_chunk = [word]
        else:
            current_chunk.append(word)

    # Flush last remaining chunk
    if current_chunk:
        cue_start = current_chunk[0].start
        cue_end = current_chunk[-1].end
        if cue_end <= cue_start:
            cue_end = cue_start + 0.1

        cue_text = wrap_cue_text(
            [w.word for w in current_chunk],
            max_line_length=max_line_length,
            max_lines=max_lines_per_subtitle,
        )

        cues.append(
            SubtitleBlock(
                index=start_index + len(cues),
                start_time=cue_start,
                end_time=cue_end,
                text=cue_text,
            )
        )

    return cues


def segment_transcription_result(
    result: Sequence[TranscriptionSegment],
    max_line_length: int = 42,
    max_lines_per_subtitle: int = 2,
    max_pause_seconds: float = 0.6,
    max_cue_duration: float = 6.0,
) -> List[SubtitleBlock]:
    """
    Convert a list of TranscriptionSegment objects into subtitle cues.
    Uses word-level timestamps when available; gracefully falls back to segment boundaries.

    Args:
        result: List of TranscriptionSegment instances from ASR.
        max_line_length: Max characters per line.
        max_lines_per_subtitle: Max lines per subtitle.
        max_pause_seconds: Pause threshold between words.
        max_cue_duration: Max cue duration.

    Returns:
        List of SubtitleBlock instances.
    """
    if not result:
        return []

    all_cues: List[SubtitleBlock] = []

    for seg in result:
        if seg.words:
            # Word-level segmentation
            seg_cues = segment_words_into_cues(
                words=seg.words,
                start_index=len(all_cues) + 1,
                max_line_length=max_line_length,
                max_lines_per_subtitle=max_lines_per_subtitle,
                max_pause_seconds=max_pause_seconds,
                max_cue_duration=max_cue_duration,
            )
            all_cues.extend(seg_cues)
        else:
            # Fallback to segment-level timestamp
            logger.warning(
                "Segment #%d lacks word-level timestamps; falling back to segment interval (%.2fs - %.2fs).",
                seg.id, seg.start, seg.end
            )
            cue_start = seg.start
            cue_end = seg.end
            if cue_end <= cue_start:
                cue_end = cue_start + 0.5

            wrapped = wrap_cue_text(
                seg.text.split(),
                max_line_length=max_line_length,
                max_lines=max_lines_per_subtitle,
            )

            all_cues.append(
                SubtitleBlock(
                    index=len(all_cues) + 1,
                    start_time=cue_start,
                    end_time=cue_end,
                    text=wrapped,
                )
            )

    return all_cues


def segment_into_subtitles(
    words: Sequence[WordTiming],
    max_line_length: int = 42,
    max_lines_per_subtitle: int = 2,
    max_pause_seconds: float = 0.6,
    max_cue_duration: float = 6.0,
) -> List[SubtitleBlock]:
    """Convenience wrapper for raw WordTiming lists."""
    return segment_words_into_cues(
        words=words,
        start_index=1,
        max_line_length=max_line_length,
        max_lines_per_subtitle=max_lines_per_subtitle,
        max_pause_seconds=max_pause_seconds,
        max_cue_duration=max_cue_duration,
    )
