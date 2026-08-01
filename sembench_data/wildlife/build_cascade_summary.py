#!/usr/bin/env python3
"""Build the Wildlife DASE cascade summary."""

import sys
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCENARIO_DIR.parent))

from dase_cascade.summary import build_cascade_summary  # noqa: E402


QORDER = [str(query_id) for query_id in range(1, 11)]
OPERATORS = {
    "1": "F",
    "2": "F",
    "3": "F",
    "4": "F",
    "5": "J",
    "6": "J",
    "7": "J",
    "8": "J",
    "9": "J",
    "10": "F+G",
}


def main() -> None:
    build_cascade_summary(SCENARIO_DIR, QORDER, OPERATORS)


if __name__ == "__main__":
    main()