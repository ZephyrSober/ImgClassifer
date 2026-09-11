from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONVERTER = REPO_ROOT / "converter" / "converter" / "converter_lite.exe"


def _add_converter_lib_to_path(converter_path: str) -> None:
   
    converter = Path(converter_path).resolve()
    lib_dir = converter.parent.parent / "lib"
    if lib_dir.is_dir():
        os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MindIR model to a MindSpore Lite .ms model via converter_lite."
    )
    parser.add_argument(
        "--converter",
        default=str(DEFAULT_CONVERTER),
        help="Path to converter_lite.exe. Defaults to the converter bundled in this repo.",
    )
    parser.add_argument("--model", default="shufflenetv2.mindir", help="Path to the .mindir file.")
    parser.add_argument("--output", default="pet", help="Output .ms name without extension.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not Path(args.converter).exists():
        raise FileNotFoundError(
            f"converter_lite not found at {args.converter}. "
            "Pass --converter to point at your converter_lite.exe."
        )
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"MindIR model not found: {args.model}. Run from the repo root or pass --model."
        )

    _add_converter_lib_to_path(args.converter)

    command = [
        args.converter,
        "--fmk=MINDIR",
        f"--modelFile={args.model}",
        f"--outputFile={args.output}",
    ]
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
