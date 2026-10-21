"""
Command Line Interface (CLI) Dispatcher for Speech SRT Tool.

Enables headless pipeline execution:
python -m speech_srt.cli --input "video.mp4" --output "output.srt" --model small.en --device auto
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from speech_srt.config import AppConfig, load_config
from speech_srt.pipeline import TranscriptionPipeline

# Terminal formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def create_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m speech_srt.cli",
        description="Speech SRT Tool -- High-Accuracy Offline English Subtitle Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to source video or audio file (.mp4, .mkv, .mov, .wav, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Destination path for the exported .srt file (defaults to output/<filename>.srt)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="small.en",
        help="Whisper model size (e.g., 'tiny.en', 'base.en', 'small.en', 'medium.en')",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute hardware target ('auto' safely detects CUDA, falling back to CPU)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Quantization precision ('int8', 'float16', 'float32'). Auto-selected by default.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for transcription decoding",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Silero Voice Activity Detector (VAD) filter",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Enforce strict offline mode (prevent automatic model weight downloads)",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=42,
        help="CapCut maximum characters per subtitle line",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """Execute CLI workflow."""
    parser = create_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    print(f"\n{BOLD}{CYAN}==================================================================={RESET}")
    print(f"{BOLD}{CYAN}             Speech SRT Tool -- Command Line Pipeline              {RESET}")
    print(f"{BOLD}{CYAN}==================================================================={RESET}")
    print(f"  Input Media  : {input_path}")
    print(f"  Output SRT   : {output_path or '(Default output directory)'}")
    print(f"  Model Size   : {args.model}")
    print(f"  Device Target: {args.device} (compute_type: {args.compute_type or 'auto'})")
    print(f"  VAD Filter   : {'Enabled' if args.vad else 'Disabled'}")
    print(f"  Offline Mode : {'Active' if args.offline else 'Permit model download'}")
    print(f"{BOLD}{CYAN}-------------------------------------------------------------------{RESET}\n")

    # Load and adjust base config
    config: AppConfig = load_config()
    config.subtitles.max_line_length = args.max_line_length

    pipeline = TranscriptionPipeline(
        config=config,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        offline=args.offline,
    )

    start_time = time.time()

    def on_progress(percentage: float, message: str) -> None:
        elapsed = time.time() - start_time
        print(f"  {CYAN}[{percentage:5.1f}%]{RESET} ({elapsed:5.1f}s) {message}")

    result = pipeline.process(
        input_file=input_path,
        output_srt_file=output_path,
        progress_callback=on_progress,
        beam_size=args.beam_size,
        vad_filter=args.vad,
    )

    total_time = time.time() - start_time

    print(f"\n{BOLD}{CYAN}-------------------------------------------------------------------{RESET}")
    if result.success:
        print(f"{BOLD}{GREEN}  STATUS: SUCCESS!{RESET}")
        print(f"  Exported SRT File : {result.output_srt_file}")
        print(f"  Subtitle Blocks   : {result.subtitle_count} cues")
        print(f"  Media Duration    : {result.duration_seconds:.2f} seconds")
        print(f"  Total Run Time    : {total_time:.2f} seconds")

        if result.warnings:
            print(f"\n{YELLOW}{BOLD}  Warnings ({len(result.warnings)}):{RESET}")
            for w in result.warnings:
                print(f"    - {w}")

        print(f"{BOLD}{CYAN}==================================================================={RESET}\n")
        return 0
    else:
        print(f"{BOLD}{RED}  STATUS: FAILED!{RESET}")
        print(f"  Error Details     : {result.error_message}")
        print(f"  Elapsed Time      : {total_time:.2f} seconds")
        print(f"{BOLD}{CYAN}==================================================================={RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
