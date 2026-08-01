#!/usr/bin/env python3
"""Build the MMQA DASE cascade summary."""

import sys
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCENARIO_DIR.parent))

from dase_cascade.summary import build_cascade_summary  # noqa: E402


QORDER = ["1", "2a", "2b", "3a", "3f", "4", "5", "6a", "6b", "6c", "7"]
OPERATORS = {
    "1": "M",
    "2a": "J",
    "2b": "J",
    "3a": "F",
    "3f": "F",
    "4": "M",
    "5": "M",
    "6a": "F",
    "6b": "F",
    "6c": "F",
    "7": "J",
}


def main() -> None:
    build_cascade_summary(SCENARIO_DIR, QORDER, OPERATORS)


if __name__ == "__main__":
    main()