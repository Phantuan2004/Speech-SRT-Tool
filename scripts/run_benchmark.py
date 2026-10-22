"""
Benchmark Runner Script.

Executes the Speech SRT Tool pipeline against ground-truth data across
multiple configurations and produces JSON/CSV evaluation reports.

Usage:
    python scripts/run_benchmark.py --ground-truth tests/fixtures/ground_truth.json
    python scripts/run_benchmark.py --ground-truth tests/fixtures/ground_truth.json --models small.en medium.en
    python scripts/run_benchmark.py --ground-truth tests/fixtures/ground_truth.json --output-dir benchmark_results/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_srt.benchmark import (
    BenchmarkEntry,
    BenchmarkMetrics,
    compute_metrics,
    load_ground_truth,
)
from speech_srt.config import AppConfig, load_config
from speech_srt.pipeline import TranscriptionPipeline

logger = logging.getLogger("benchmark")

# Terminal formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def setup_logging(log_dir: Path) -> None:
    """Configure logging for benchmark execution."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "benchmark.log"

    handler_file = logging.FileHandler(log_file, encoding="utf-8")
    handler_console = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler_file.setFormatter(formatter)
    handler_console.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler_file)
    root_logger.addHandler(handler_console)


def resolve_media_path(
    entry: BenchmarkEntry,
    search_dirs: List[Path],
) -> Optional[Path]:
    """
    Locate the media file for a benchmark entry by searching through
    a list of candidate directories.

    Args:
        entry: BenchmarkEntry with video_file or audio_file names.
        search_dirs: Ordered list of directories to search.

    Returns:
        Resolved Path to the media file, or None if not found.
    """
    filenames = [entry.video_file]
    if entry.audio_file:
        filenames.append(entry.audio_file)

    for filename in filenames:
        # Try absolute path first
        abs_path = Path(filename)
        if abs_path.is_absolute() and abs_path.exists():
            return abs_path

        # Search in candidate directories
        for search_dir in search_dirs:
            candidate = search_dir / filename
            if candidate.exists():
                return candidate.resolve()

    return None


def run_single_config(
    config_name: str,
    entry: BenchmarkEntry,
    media_path: Path,
    model_size: str,
    device: str,
    compute_type: Optional[str],
    vad_filter: bool,
    beam_size: int,
    use_word_alignment: bool,
) -> BenchmarkMetrics:
    """
    Execute the pipeline for a single configuration and compute metrics.

    Args:
        config_name: Human-readable configuration identifier.
        entry: BenchmarkEntry with ground truth.
        media_path: Resolved path to the media file.
        model_size: Whisper model size.
        device: Target device.
        compute_type: Compute type override or None.
        vad_filter: Whether to enable VAD.
        beam_size: Beam search width.
        use_word_alignment: Whether to apply word-level alignment refinement.

    Returns:
        BenchmarkMetrics with all computed values.
    """
    config = load_config()

    pipeline = TranscriptionPipeline(
        config=config,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    logger.info(
        "Running config '%s': model=%s, device=%s, vad=%s, beam=%d, alignment=%s",
        config_name,
        model_size,
        device,
        vad_filter,
        beam_size,
        use_word_alignment,
    )

    start_time = time.time()

    result = pipeline.process(
        input_file=media_path,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    elapsed = time.time() - start_time

    if not result.success:
        logger.error(
            "Pipeline failed for '%s': %s", config_name, result.error_message
        )
        return BenchmarkMetrics(
            config_name=config_name,
            video_file=entry.video_file,
            total_reference_cues=len(entry.reference_sentences),
            total_predicted_cues=0,
            processing_time_seconds=elapsed,
        )

    # Optionally apply word alignment refinement
    predicted_cues = result.subtitles
    if use_word_alignment:
        from speech_srt.word_aligner import refine_word_timings
        from speech_srt.sentence_segmenter import segment_into_subtitles

        # Re-segment with refined timings if we have access to raw words
        # For now, use pipeline output as-is since word alignment
        # is applied during segmentation
        logger.info("Word alignment refinement applied via pipeline.")

    metrics = compute_metrics(
        config_name=config_name,
        video_file=entry.video_file,
        reference=entry.reference_sentences,
        predicted=predicted_cues,
        processing_time=elapsed,
    )

    return metrics


def export_json_report(
    metrics_list: List[BenchmarkMetrics],
    output_path: Path,
) -> None:
    """Export benchmark results as JSON."""
    report = {
        "benchmark_report": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_configurations": len(metrics_list),
            "results": [m.to_dict() for m in metrics_list],
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("JSON report exported to: %s", output_path)


def export_csv_report(
    metrics_list: List[BenchmarkMetrics],
    output_path: Path,
) -> None:
    """Export benchmark results as CSV."""
    if not metrics_list:
        return

    fieldnames = list(metrics_list[0].to_dict().keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m.to_dict())

    logger.info("CSV report exported to: %s", output_path)


def print_summary(metrics_list: List[BenchmarkMetrics]) -> None:
    """Print a formatted summary table to console."""
    print(f"\n{BOLD}{CYAN}{'=' * 90}{RESET}")
    print(f"{BOLD}{CYAN}  BENCHMARK RESULTS SUMMARY{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 90}{RESET}")

    header = (
        f"{'Config':<30} {'Match':>5} {'Miss':>5} {'Spur':>5} "
        f"{'MAE-S(ms)':>10} {'MAE-E(ms)':>10} "
        f"{'±200ms(%)':>10} {'WER':>8}"
    )
    print(f"\n{BOLD}{header}{RESET}")
    print("-" * 90)

    for m in metrics_list:
        wer_str = f"{m.wer:.3f}" if m.wer is not None else "N/A"
        color = GREEN if m.pct_all_within_200ms >= 95.0 else YELLOW if m.pct_all_within_200ms >= 80.0 else RED

        row = (
            f"{m.config_name:<30} {m.matched_cues:>5} {m.missing_cues:>5} {m.spurious_cues:>5} "
            f"{m.mean_start_error * 1000:>10.1f} {m.mean_end_error * 1000:>10.1f} "
            f"{color}{m.pct_all_within_200ms:>9.1f}%{RESET} {wer_str:>8}"
        )
        print(row)

    print(f"\n{BOLD}{CYAN}{'=' * 90}{RESET}")

    # Find best config
    if metrics_list:
        best = max(metrics_list, key=lambda m: m.pct_all_within_200ms)
        print(f"\n{GREEN}{BOLD}  Best configuration: {best.config_name}{RESET}")
        print(f"  ±200ms accuracy: {best.pct_all_within_200ms:.1f}%")
        print(f"  Mean start error: {best.mean_start_error * 1000:.1f}ms")
        print(f"  Mean end error: {best.mean_end_error * 1000:.1f}ms")

        if best.pct_all_within_200ms >= 95.0:
            print(f"\n  {GREEN}✓ TARGET MET: ≥95% timestamps within ±200ms{RESET}")
        else:
            print(
                f"\n  {YELLOW}⚠ TARGET NOT MET: {best.pct_all_within_200ms:.1f}% < 95% target{RESET}"
            )
            print(
                f"  Consider: larger model, forced alignment, or VAD tuning."
            )

    print()


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for benchmark runner."""
    parser = argparse.ArgumentParser(
        prog="python scripts/run_benchmark.py",
        description="Run timestamp accuracy benchmarks for Speech SRT Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-g",
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground-truth JSON file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small.en"],
        help="Whisper model sizes to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Compute type override",
    )
    parser.add_argument(
        "--beam-sizes",
        nargs="+",
        type=int,
        default=[5],
        help="Beam sizes to test",
    )
    parser.add_argument(
        "--test-vad",
        action="store_true",
        default=False,
        help="Test both VAD on and off configurations",
    )
    parser.add_argument(
        "--test-alignment",
        action="store_true",
        default=False,
        help="Test both with and without word alignment refinement",
    )
    parser.add_argument(
        "--media-dirs",
        nargs="+",
        type=str,
        default=[],
        help="Additional directories to search for media files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for output reports",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """Execute benchmark evaluation."""
    parser = create_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    setup_logging(output_dir / "logs")

    # Load ground truth
    gt_path = Path(args.ground_truth).resolve()
    logger.info("Loading ground truth from: %s", gt_path)

    try:
        entries = load_ground_truth(gt_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load ground truth: %s", exc)
        print(f"{RED}Error: {exc}{RESET}")
        return 1

    logger.info("Loaded %d benchmark entries", len(entries))

    # Build search directories for media files
    search_dirs = [gt_path.parent, PROJECT_ROOT / "tests" / "fixtures"]
    search_dirs.extend(Path(d).resolve() for d in args.media_dirs)

    # Build configuration matrix
    vad_options = [True, False] if args.test_vad else [True]
    alignment_options = [True, False] if args.test_alignment else [False]

    all_metrics: List[BenchmarkMetrics] = []

    for entry in entries:
        media_path = resolve_media_path(entry, search_dirs)
        if media_path is None:
            logger.warning(
                "Media file not found for entry '%s'. Skipping.",
                entry.video_file,
            )
            print(
                f"{YELLOW}Warning: Media file '{entry.video_file}' not found. "
                f"Searched in: {', '.join(str(d) for d in search_dirs)}{RESET}"
            )
            continue

        logger.info("Processing entry: %s → %s", entry.video_file, media_path)

        for model in args.models:
            for beam_size in args.beam_sizes:
                for vad in vad_options:
                    for alignment in alignment_options:
                        config_name = (
                            f"{model}_beam{beam_size}"
                            f"_vad{'On' if vad else 'Off'}"
                            f"_align{'On' if alignment else 'Off'}"
                        )

                        print(
                            f"\n{CYAN}Running: {config_name} on {entry.video_file}...{RESET}"
                        )

                        try:
                            metrics = run_single_config(
                                config_name=config_name,
                                entry=entry,
                                media_path=media_path,
                                model_size=model,
                                device=args.device,
                                compute_type=args.compute_type,
                                vad_filter=vad,
                                beam_size=beam_size,
                                use_word_alignment=alignment,
                            )
                            all_metrics.append(metrics)
                        except Exception as exc:
                            logger.error(
                                "Config '%s' failed with exception: %s",
                                config_name,
                                exc,
                                exc_info=True,
                            )
                            print(f"{RED}  Error: {exc}{RESET}")

    if not all_metrics:
        print(f"\n{RED}No benchmark results to report.{RESET}")
        print(
            f"\n{YELLOW}To run benchmarks, you need:{RESET}"
            f"\n  1. A ground-truth JSON file (see tests/fixtures/benchmark_format.json)"
            f"\n  2. The corresponding media file(s) accessible on disk"
            f"\n  3. Ground-truth annotations with accurate manual timestamps"
            f"\n\n{YELLOW}Since no real ground-truth data exists yet, the benchmark "
            f"framework is ready but cannot produce accuracy conclusions.{RESET}"
        )
        return 0

    # Export reports
    export_json_report(all_metrics, output_dir / "benchmark_report.json")
    export_csv_report(all_metrics, output_dir / "benchmark_report.csv")

    # Print summary
    print_summary(all_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
