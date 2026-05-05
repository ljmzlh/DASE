"""Per-row cost calibration via BQ AI.GENERATE_BOOL proxy.

Every existing script ships a ~50-line `per_row_cost_calibration(...)` that
samples K rows, runs AI.GENERATE_BOOL with the same prompt + same modality
binding the actual operator uses, and accounts tokens by category
(input_other / input_audio / output / thoughts).

Centralized here so each cascade script just calls per_row_cost(...).
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

from google.cloud import bigquery

from dase_cascade.runtime import PRICES


@dataclass
class CalibrationResult:
    method: str
    n_sample: int
    tokens_total: dict           # {prompt_other, prompt_audio, output, thoughts}
    sample_cost_usd: float
    per_row_cost_usd: float
    elapsed_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def _sum_tokens(verdicts) -> tuple:
    """Aggregate token counts across a verdict batch (BQ AI.GENERATE_BOOL output).
    Returns (input_other, input_audio, output, thoughts)."""
    p_other = p_audio = out = thoughts = 0
    for v in verdicts:
        um = json.loads(v["full_response"]).get("usage_metadata", {})
        out += um.get("candidates_token_count", 0) or 0
        thoughts += um.get("thoughts_token_count", 0) or 0
        details = um.get("prompt_tokens_details") or []
        a = sum(d.get("token_count", 0) for d in details if d.get("modality") == "AUDIO")
        o = sum(d.get("token_count", 0) for d in details if d.get("modality") != "AUDIO")
        if not details:
            o = um.get("prompt_token_count", 0) or 0
        p_audio += a
        p_other += o
    return p_other, p_audio, out, thoughts


def _to_cost(p_other: int, p_audio: int, out: int, thoughts: int) -> float:
    return (
        p_other * PRICES["input_other"]
        + p_audio * PRICES["input_audio"]
        + (out + thoughts) * PRICES["output"]
    )


def per_row_cost(
    client: bigquery.Client,
    prompt: str,
    *,
    sample_uris: Optional[List[str]] = None,
    sample_texts: Optional[List[str]] = None,
    ext_table: Optional[str] = None,           # required if sample_uris given
    text_from_table_sql: Optional[str] = None, # for text-from-existing-table style (movie Q1)
    method_label: Optional[str] = None,
    k: int = 10,
) -> CalibrationResult:
    """Calibrate per-row cost for a BQ semantic-op invocation.

    Three call modes, mirroring how existing scripts calibrate:

      (1) image/audio: pass `sample_uris=[...]` + `ext_table` (e.g.
          'animals_dataset.image_data_external'). Each row binds ot.uri = uri_i.
      (2) text inline: pass `sample_texts=[...]`. Each row binds @text_i.
      (3) text from existing BQ table: pass `text_from_table_sql=
          'movie.reviews AS r' + col 'r.reviewText'`. We then build:
            SELECT AI.GENERATE_BOOL((prompt, r.reviewText), …) FROM movie.reviews AS r LIMIT k
          Tip: stick `prompt = "<prompt>', r.reviewText"` syntax in caller.
          Easier: caller passes the full `text_from_table_sql` (a SELECT … LIMIT k).

    Always uses thinking_budget=0 so the proxy reports a tight lower bound.
    """
    rows_sql, params, mode = _build_calibration_sql(
        prompt, sample_uris, sample_texts, ext_table, text_from_table_sql, k
    )
    cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=False)
    t0 = time.time()
    df = client.query(rows_sql, job_config=cfg).result().to_dataframe()
    elapsed = time.time() - t0
    p_other, p_audio, out, thoughts = _sum_tokens(df["verdict"])
    n = len(df)
    cost = _to_cost(p_other, p_audio, out, thoughts)
    return CalibrationResult(
        method=method_label or f"AI.GENERATE_BOOL ({mode}) + thinking_budget=0",
        n_sample=n,
        tokens_total={"prompt_other": p_other, "prompt_audio": p_audio,
                      "output": out, "thoughts": thoughts},
        sample_cost_usd=cost,
        per_row_cost_usd=cost / n if n else 0.0,
        elapsed_s=elapsed,
    )


def _build_calibration_sql(prompt, sample_uris, sample_texts, ext_table, text_from_table_sql, k):
    """Pick mode and produce (sql, parameters, mode_label)."""
    THINKING = "JSON '{\"generation_config\":{\"thinking_config\":{\"thinking_budget\":0}}}'"
    if text_from_table_sql is not None:
        # caller-provided "FROM <table> WHERE <col>" inline; we wrap with AI.GENERATE_BOOL
        # text_from_table_sql is expected to be a complete SELECT … LIMIT k that returns
        # one column named "verdict" of type STRUCT<… full_response …>. For generality we
        # let caller hand us the raw SQL.
        return text_from_table_sql, [], "text_from_table"
    if sample_uris is not None:
        if not ext_table:
            raise ValueError("sample_uris requires ext_table=")
        selects = []
        params = []
        for i, uri in enumerate(sample_uris[:k]):
            selects.append(
                f"\nSELECT AI.GENERATE_BOOL("
                f"\n  ('{prompt}', ot.ref),"
                f"\n  connection_id => 'us.connection',"
                f"\n  endpoint => 'gemini-2.5-flash',"
                f"\n  model_params => {THINKING}"
                f"\n) AS verdict"
                f"\nFROM {ext_table} ot"
                f"\nWHERE ot.uri = @uri_{i}"
            )
            params.append(bigquery.ScalarQueryParameter(f"uri_{i}", "STRING", uri))
        return " UNION ALL ".join(selects), params, "uri+ext_table"
    if sample_texts is not None:
        selects = []
        params = []
        for i, t in enumerate(sample_texts[:k]):
            selects.append(
                f"\nSELECT AI.GENERATE_BOOL("
                f"\n  ('{prompt}', @text_{i}),"
                f"\n  connection_id => 'us.connection',"
                f"\n  endpoint => 'gemini-2.5-flash',"
                f"\n  model_params => {THINKING}"
                f"\n) AS verdict"
            )
            params.append(bigquery.ScalarQueryParameter(f"text_{i}", "STRING", t))
        return " UNION ALL ".join(selects), params, "inline_text"
    raise ValueError("must pass one of sample_uris, sample_texts, text_from_table_sql")
