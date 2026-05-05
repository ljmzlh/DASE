"""
Workload loader: 读取 workload 目录下的 JSON，供 baseline 执行使用.
Query 文件中 semantic 信号可含 query_embed（1536 维），由 fill_embeddings 填入.
Usage:
  from workload.load import load_workload, get_queries
  data = load_workload("workload/w6_queries_example.json")
  for q in get_queries(data): ...
"""
import json
import os
from typing import Any, Dict, List


def load_workload(path: str) -> Dict[str, Any]:
    """加载单个 workload 文件，返回完整 dict（含 workload, queries 等）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_queries(workload_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 load_workload 返回的 dict 中取出 query 列表。"""
    return workload_dict.get("queries", [])


def load_w6_queries(path: str = None) -> List[Dict[str, Any]]:
    """加载 W6 所有 query；path 默认 workload/w6_queries_example.json。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "w6_queries_example.json")
    data = load_workload(path)
    return get_queries(data)
