"""BqVerifier — runs the LLM-stage of a cascade on uncertain rows.

Two flavors mirror the BQ semantic operators used in sembench:

  AiIfVerifier        AI.IF  (boolean filter, returns subset of ids)
  AiGenerateVerifier  AI.GENERATE (free-form text per row, returns id→value map)

Each verifier:
  1. Optionally CTAS a small staging table from the candidate set.
  2. Runs the BQ semantic-op SQL.
  3. Returns a VerifierResult with positive ids / generated values + cost stats.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from google.cloud import bigquery

from dase_cascade.runtime import run_query


@dataclass
class VerifierResult:
    """Outcome of one BQ verifier invocation.

    For AiIfVerifier: `positive_ids` is the subset that returned TRUE; `values` is empty.
    For AiGenerateVerifier: `values` maps id → generated string; `positive_ids` is its keys.
    """
    positive_ids: set                  # subset of input candidate ids that BQ returned
    values: Dict[Any, Any] = field(default_factory=dict)
    n_calls: int = 0                   # number of LLM rows actually processed
    cost_usd: float = 0.0
    wall_s: float = 0.0
    slot_ms: int = 0
    sql: str = ""
    ctas_wall_s: float = 0.0           # CTAS staging-table latency, if any
    ctas_slot_ms: int = 0
    ctas_sql: str = ""

    def to_dict(self) -> dict:
        return {
            "positive_ids": sorted(self.positive_ids) if self.positive_ids else [],
            "n_returned_values": len(self.values),
            "n_calls": self.n_calls,
            "cost_usd": self.cost_usd,
            "wall_s": self.wall_s,
            "slot_ms": self.slot_ms,
            "ctas_wall_s": self.ctas_wall_s,
            "ctas_slot_ms": self.ctas_slot_ms,
        }


class BqVerifier(ABC):
    """Abstract verifier; subclasses know how to issue a BQ semantic-op call."""

    @abstractmethod
    def verify(self, client: bigquery.Client, candidate_ids: Sequence[Any],
               per_row_cost: float) -> VerifierResult:
        ...


@dataclass
class AiIfVerifier(BqVerifier):
    """Run AI.IF on a candidate set; return ids that BQ said TRUE.

    Two ways to scope the AI.IF call:
      * Provide `make_staging_sql=fn(ids)` + `verify_sql` referencing a staging table.
      * Or provide just `verify_sql_template=fn(ids)` that inlines an IN(...) clause.

    The first is preferred for image/audio cascades that need a CTAS join with
    EXTERNAL_OBJECT_TRANSFORM; the second is fine for text-only filters.

    `id_column` is the column name of the candidate id in BQ output rows.
    """
    verify_sql: Optional[str] = None
    verify_sql_template: Optional[Callable[[Sequence[Any]], str]] = None
    make_staging_sql: Optional[Callable[[Sequence[Any]], str]] = None
    id_column: str = "id"
    coerce_id: Callable[[Any], Any] = int

    def verify(self, client, candidate_ids, per_row_cost) -> VerifierResult:
        ids = list(candidate_ids)
        if not ids:
            return VerifierResult(positive_ids=set())

        ctas_wall_s = 0.0
        ctas_slot_ms = 0
        ctas_sql = ""
        if self.make_staging_sql is not None:
            ctas_sql = self.make_staging_sql(ids)
            _, ctas_wall_s, ctas_slot_ms, _ = run_query(client, ctas_sql)

        if self.verify_sql is not None:
            sql = self.verify_sql
        elif self.verify_sql_template is not None:
            sql = self.verify_sql_template(ids)
        else:
            raise ValueError("AiIfVerifier needs verify_sql or verify_sql_template")

        df, wall, slot, _ = run_query(client, sql)
        pos = {self.coerce_id(x) for x in df[self.id_column]}
        n_calls = len(ids)
        return VerifierResult(
            positive_ids=pos, n_calls=n_calls, cost_usd=per_row_cost * n_calls,
            wall_s=wall, slot_ms=slot, sql=sql,
            ctas_wall_s=ctas_wall_s, ctas_slot_ms=ctas_slot_ms, ctas_sql=ctas_sql,
        )


@dataclass
class AiGenerateVerifier(BqVerifier):
    """Run AI.GENERATE on a candidate set; return id→generated value map.

    Used by ecomm Q3 (brand extraction) and similar SEM_MAP queries where the
    BQ stage produces a string per row.
    """
    verify_sql: Optional[str] = None
    verify_sql_template: Optional[Callable[[Sequence[Any]], str]] = None
    make_staging_sql: Optional[Callable[[Sequence[Any]], str]] = None
    id_column: str = "id"
    value_column: str = "category"
    coerce_id: Callable[[Any], Any] = int
    coerce_value: Callable[[Any], Any] = lambda v: str(v).strip()  # noqa: E731

    def verify(self, client, candidate_ids, per_row_cost) -> VerifierResult:
        ids = list(candidate_ids)
        if not ids:
            return VerifierResult(positive_ids=set())
        ctas_wall_s = 0.0
        ctas_slot_ms = 0
        ctas_sql = ""
        if self.make_staging_sql is not None:
            ctas_sql = self.make_staging_sql(ids)
            _, ctas_wall_s, ctas_slot_ms, _ = run_query(client, ctas_sql)

        if self.verify_sql is not None:
            sql = self.verify_sql
        elif self.verify_sql_template is not None:
            sql = self.verify_sql_template(ids)
        else:
            raise ValueError("AiGenerateVerifier needs verify_sql or verify_sql_template")

        df, wall, slot, _ = run_query(client, sql)
        values = {
            self.coerce_id(row[self.id_column]): self.coerce_value(row[self.value_column])
            for _, row in df.iterrows()
        }
        n_calls = len(ids)
        return VerifierResult(
            positive_ids=set(values.keys()), values=values,
            n_calls=n_calls, cost_usd=per_row_cost * n_calls,
            wall_s=wall, slot_ms=slot, sql=sql,
            ctas_wall_s=ctas_wall_s, ctas_slot_ms=ctas_slot_ms, ctas_sql=ctas_sql,
        )
