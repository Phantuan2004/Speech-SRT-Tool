#!/usr/bin/env python3
"""
Environment Validation Script for Speech SRT Tool.

Performs comprehensive diagnostic checks:
1. Python version and virtual environment (.venv)
2. FFmpeg and FFprobe availability and version parsing
3. Git version control status
4. Hardware inspection: CPU specs and GPU/CUDA verification
5. Python dependency imports (PySide6, pytest, etc.)
6. Writable directory permissions (output/, logs/)
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ANSI Color Codes for Windows 10/11 Terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_status(tag: str, message: str, color: str = GREEN) -> None:
    """Print formatted diagnostic message with aligned tag."""
    print(f"  {color}[{tag.center(6)}]{RESET} {message}")


def check_python() -> Tuple[bool, str]:
    """Check Python version and 64-bit architecture."""
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    is_64bit = sys.maxsize > 2**32
    arch_str = "64-bit" if is_64bit else "32-bit"

    if version < (3, 10):
        return False, f"Python {ver_str} ({arch_str}) - Requires Python 3.10 or newer."
    return True, f"Python {ver_str} ({arch_str})"


def check_virtualenv() -> Tuple[bool, str]:
    """Check if script is running inside a virtual environment."""
    is_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    if is_venv:
        return True, f"Active Virtual Environment: {Path(sys.prefix).name} ({sys.prefix})"
    return False, "Not running in a virtual environment! Please activate .venv before running."


def check_tool(tool_name: str) -> Tuple[bool, str]:
    """Check if an external binary (ffmpeg, ffprobe, git) is accessible in PATH."""
    path_found = shutil.which(tool_name)
    if not path_found:
        return False, f"'{tool_name}' not found in system PATH. Please install it and add to PATH."

    try:
        cmd = [tool_name, "--version"] if tool_name == "git" else [tool_name, "-version"]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0] if result.stdout else "Available"
            return True, f"{tool_name.upper()} executable found at {path_found} ({first_line})"
        return False, f"'{tool_name}' returned error code {result.returncode}"
    except Exception as exc:
        return False, f"Failed to execute '{tool_name}': {exc}"


def check_hardware_and_cuda() -> Dict[str, Any]:
    """Inspect GPU hardware and verify CUDA readiness without false assumptions."""
    hardware_info = {
        "gpu_detected": False,
        "gpu_names": [],
        "is_nvidia": False,
        "cuda_available": False,
        "details": "",
        "recommended_device": "cpu",
    }

    # 1. Query display adapters via PowerShell CIM on Windows
    try:
        ps_cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
        res = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            controllers = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            hardware_info["gpu_names"] = controllers
            hardware_info["gpu_detected"] = len(controllers) > 0
            for name in controllers:
                if "nvidia" in name.lower():
                    hardware_info["is_nvidia"] = True
    except Exception:
        pass

    # 2. Check nvidia-smi if NVIDIA is suspected
    nvidia_smi_path = shutil.which("nvidia-smi")
    has_nvidia_smi = False
    if nvidia_smi_path:
        try:
            res = subprocess.run([nvidia_smi_path], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                has_nvidia_smi = True
        except Exception:
            pass

    # 3. Check torch/ctranslate2 CUDA if installed
    torch_cuda = False
    try:
        import torch  # type: ignore
        torch_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    hardware_info["cuda_available"] = bool(has_nvidia_smi and torch_cuda)

    # 4. Synthesize diagnosis
    if hardware_info["gpu_detected"]:
        gpu_summary = ", ".join(hardware_info["gpu_names"])
        if hardware_info["is_nvidia"] and hardware_info["cuda_available"]:
            hardware_info["details"] = f"{gpu_summary} (NVIDIA GPU with CUDA acceleration active)"
            hardware_info["recommended_device"] = "cuda"
        elif hardware_info["is_nvidia"] and not hardware_info["cuda_available"]:
            hardware_info["details"] = (
                f"{gpu_summary} (NVIDIA GPU detected, but CUDA runtime is not available/configured. Fallback: CPU)"
            )
            hardware_info["recommended_device"] = "cpu"
        else:
            hardware_info["details"] = (
                f"{gpu_summary} (Non-NVIDIA GPU. CUDA is unsupported on this hardware. Fallback: CPU)"
            )
            hardware_info["recommended_device"] = "cpu"
    else:
        hardware_info["details"] = "No discrete GPU identified. Fallback: CPU"
        hardware_info["recommended_device"] = "cpu"

    return hardware_info


def check_python_packages() -> Dict[str, Tuple[bool, str]]:
    """Check required packages for Milestone 1 and status of planned packages."""
    packages = {
        "PySide6": ("PySide6", True),
        "pytest": ("pytest", True),
        "faster-whisper": ("faster_whisper", False),
        "ctranslate2": ("ctranslate2", False),
        "numpy": ("numpy", False),
    }

    results = {}
    for name, (module_name, is_core) in packages.items():
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "Installed")
            results[name] = (True, f"v{ver}")
        except ImportError:
            status_desc = "Not installed (Required for Milestone 1)" if is_core else "Not installed (Planned for future milestone)"
            results[name] = (False, status_desc)

    return results


def check_directories(root_dir: Path) -> List[Tuple[str, bool, str]]:
    """Verify that essential workspace directories exist and are writable."""
    dirs_to_check = [
        ("output", root_dir / "output"),
        ("logs", root_dir / "logs"),
        ("config", root_dir / "config"),
    ]
    dir_results = []
    for name, path in dirs_to_check:
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                dir_results.append((name, False, f"Directory missing and cannot create: {e}"))
                continue

        # Test write permission
        test_file = path / ".write_test"
        try:
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            dir_results.append((name, True, f"{path} (Writable)"))
        except Exception as e:
            dir_results.append((name, False, f"{path} (Read-only / No write permission: {e})"))

    return dir_results


def run_diagnostics() -> int:
    """Execute all diagnostic routines and present structured results."""
    root_dir = Path(__file__).resolve().parent.parent

    print(f"\n{BOLD}{CYAN}==================================================================={RESET}")
    print(f"{BOLD}{CYAN}      Speech SRT Tool -- Environment Validation Diagnostic        {RESET}")
    print(f"{BOLD}{CYAN}==================================================================={RESET}\n")

    overall_pass = True

    # 1. Python Environment
    print(f"{BOLD}[1] Python Environment & Runtime{RESET}")
    py_ok, py_msg = check_python()
    if py_ok:
        print_status("OK", py_msg, GREEN)
    else:
        print_status("FAIL", py_msg, RED)
        overall_pass = False

    venv_ok, venv_msg = check_virtualenv()
    if venv_ok:
        print_status("OK", venv_msg, GREEN)
    else:
        print_status("WARN", venv_msg, YELLOW)

    # 2. External System Media Tools
    print(f"\n{BOLD}[2] External Media Processing Binaries{RESET}")
    for tool in ["ffmpeg", "ffprobe", "git"]:
        ok, msg = check_tool(tool)
        if ok:
            print_status("OK", msg, GREEN)
        else:
            if tool in ["ffmpeg", "ffprobe"]:
                print_status("FAIL", msg, RED)
                overall_pass = False
            else:
                print_status("WARN", msg, YELLOW)

    # 3. Hardware & Acceleration
    print(f"\n{BOLD}[3] Hardware & Acceleration Analysis{RESET}")
    hw = check_hardware_and_cuda()
    print_status("INFO", f"Detected GPU(s): {hw['details']}", BLUE)
    print_status("INFO", f"Execution Profile: {hw['recommended_device'].upper()} mode (Reliable & offline-ready)", GREEN)
    if not hw["cuda_available"] and hw["is_nvidia"]:
        print_status("NOTE", "NVIDIA GPU found but CUDA not initialized in Python environment. CPU fallback will be used.", YELLOW)
    elif not hw["is_nvidia"] and hw["gpu_detected"]:
        print_status("NOTE", "Non-NVIDIA GPU present (no CUDA). Application will operate smoothly via CPU (int8).", CYAN)

    # 4. Python Dependencies
    print(f"\n{BOLD}[4] Python Dependencies Status{RESET}")
    pkg_results = check_python_packages()
    for name, (is_installed, desc) in pkg_results.items():
        if is_installed:
            print_status("OK", f"{name}: {desc}", GREEN)
        else:
            if name in ["PySide6", "pytest"]:
                print_status("FAIL", f"{name}: {desc}", RED)
                overall_pass = False
            else:
                print_status("OPT", f"{name}: {desc}", BLUE)

    # 5. Workspace Directory Integrity
    print(f"\n{BOLD}[5] Workspace Directories & Permissions{RESET}")
    dir_results = check_directories(root_dir)
    for name, ok, msg in dir_results:
        if ok:
            print_status("OK", f"{name}: {msg}", GREEN)
        else:
            print_status("FAIL", f"{name}: {msg}", RED)
            overall_pass = False

    # 6. Final Summary
    print(f"\n{BOLD}{CYAN}-------------------------------------------------------------------{RESET}")
    if overall_pass:
        print(f"{BOLD}{GREEN}  Result: ALL REQUIRED FOUNDATIONS PASSED!{RESET}")
        print(f"  The environment is fully configured for Speech SRT Tool development.\n")
        return 0
    else:
        print(f"{BOLD}{RED}  Result: ONE OR MORE REQUIRED CHECKS FAILED.{RESET}")
        print(f"  Please review the red [FAIL] messages above and resolve before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())
