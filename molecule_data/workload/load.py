"""Workload loader for molecule: mirrors imdb_data.workload.load interface."""
import json
from typing import Any, Dict, List


def load_workload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_queries(workload_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    return workload_dict.get("queries", [])
