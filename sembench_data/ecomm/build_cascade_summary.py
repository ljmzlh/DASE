#!/usr/bin/env python3
"""Build the E-Commerce DASE cascade summary."""

import sys
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCENARIO_DIR.parent))

from dase_cascade.summary import build_cascade_summary  # noqa: E402


QORDER = [str(query_id) for query_id in range(1, 15)]
OPERATORS = {
    "1": "F",
    "2": "F",
    "3": "M",
    "4": "M",
    "5": "C",
    "6": "C",
    "7": "J",
    "8": "J",
    "9": "J",
    "10": "F J",
    "11": "F J",
    "12": "F M",
    "13": "F",
    "14": "F J R",
}


def main() -> None:
    build_cascade_summary(SCENARIO_DIR, QORDER, OPERATORS)


if __name__ == "__main__":
    main()