#!/usr/bin/env python3
"""Build the Movie DASE cascade summary."""

import sys
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCENARIO_DIR.parent))

from dase_cascade.summary import build_cascade_summary  # noqa: E402


QORDER = [str(query_id) for query_id in range(1, 11)]
OPERATORS = {
    "1": "F+L",
    "2": "F+L",
    "3": "F",
    "4": "F",
    "5": "J+L",
    "6": "J+L",
    "7": "J",
    "8": "C",
    "9": "R",
    "10": "R",
}


def main() -> None:
    build_cascade_summary(SCENARIO_DIR, QORDER, OPERATORS)


if __name__ == "__main__":
    main()