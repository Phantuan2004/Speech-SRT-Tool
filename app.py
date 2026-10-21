#!/usr/bin/env python3
"""
Speech SRT Tool -- Application Entrypoint.

Architectural Rule: No business logic is implemented in this file.
This file serves strictly as an application bootstrap and execution dispatcher.
"""

import sys
from pathlib import Path

# Add src/ directory to sys.path for clean package imports
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    """Bootstrap entry point for Speech SRT Tool."""
    import speech_srt

    print(f"Speech SRT Tool v{speech_srt.__version__}")
    print("Offline English Speech Recognition & CapCut Subtitle Generator\n")
    print("Project Foundation Status: Milestone 1 Active.")
    print("To validate your environment, run:")
    print("    python scripts/check_environment.py\n")
    print("To run unit tests, run:")
    print("    pytest -v\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
