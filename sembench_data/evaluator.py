"""
Standalone evaluator for movie_data q1-q10, aligned with SemBench evaluation logic.

Metric logic is copied from:
  SemBench/src/evaluator/generic_evaluator.py
  SemBench/src/scenario/movie/evaluation/evaluate.py

Ground truth is generated via DuckDB using the gold SQL queries against
the cache CSVs (cache/Reviews.csv, cache/Movies.csv).

Gold SQL (inlined from SemBench/files/movie/query/gold_sql/):
  Q1:  SELECT reviewId FROM Reviews WHERE scoreSentiment = 'POSITIVE';
  Q2:  SELECT reviewId FROM Reviews WHERE id = 'taken_3' AND scoreSentiment = 'POSITIVE';
  Q3:  SELECT COUNT(*) AS positive_review_cnt FROM Reviews WHERE id = 'taken_3' AND scoreSentiment = 'POSITIVE';
  Q4:  SELECT CAST(SUM(CASE WHEN scoreSentiment = 'POSITIVE' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) AS positivity_ratio FROM Reviews WHERE id = 'taken_3';
  Q5:  SELECT R1.id, R1.reviewId AS reviewId1, R2.reviewId AS reviewId2 FROM Reviews AS R1 JOIN Reviews AS R2 ON R1.id = R2.id AND R1.reviewId <> R2.reviewId WHERE R1.id = 'ant_man_and_the_wasp_quantumania' AND R1.scoreSentiment = R2.scoreSentiment;
  Q6:  SELECT R1.id, R1.reviewId AS reviewId1, R2.reviewId AS reviewId2 FROM Reviews AS R1 JOIN Reviews AS R2 ON R1.id = R2.id AND R1.reviewId <> R2.reviewId WHERE R1.id = 'ant_man_and_the_wasp_quantumania' AND R1.scoreSentiment <> R2.scoreSentiment;
  Q7:  SELECT R1.id, R1.reviewId AS reviewId1, R2.reviewId AS reviewId2 FROM Reviews AS R1 JOIN Reviews AS R2 ON R1.id = R2.id AND R1.reviewId <> R2.reviewId WHERE R1.id = 'ant_man_and_the_wasp_quantumania' AND R1.scoreSentiment <> R2.scoreSentiment;
  Q8:  SELECT scoreSentiment, COUNT(*) AS count FROM Reviews WHERE id = 'taken_3' GROUP BY scoreSentiment;
  Q9:  SELECT reviewId, CAST(SPLIT_PART(originalScore, '/', 1) AS FLOAT) AS reviewScore FROM Reviews WHERE id = 'ant_man_and_the_wasp_quantumania';
  Q10: SELECT M.id AS movieId, M.audienceScore AS movieScore FROM Movies AS M;
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "movie" / "cache"


# ── Metric dataclasses (aligned with SemBench) ────────────────────────────────
@dataclass
class QueryMetricRetrieval:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class QueryMetricAggregation:
    relative_error: float = 0.0
    absolute_error: float = 0.0
    mean_absolute_percentage_error: float = 0.0


@dataclass
class QueryMetricRank:
    spearman_correlation: float = 0.0
    kendall_tau: float = 0.0


# ── Ground truth via pandas ───────────────────────────────────────────────────
def _load_reviews() -> pd.DataFrame:
    return pd.read_csv(CACHE_DIR / "Reviews.csv")

def _load_movies() -> pd.DataFrame:
    return pd.read_csv(CACHE_DIR / "Movies.csv")

def get_ground_truth(query_id: int) -> pd.DataFrame:
    """Compute ground truth using pandas (equivalent to the gold SQL)."""
    r = _load_reviews()
    if query_id == 1:
        return r.loc[r["scoreSentiment"] == "POSITIVE", ["reviewId"]].reset_index(drop=True)
    if query_id == 2:
        return r.loc[(r["id"] == "taken_3") & (r["scoreSentiment"] == "POSITIVE"), ["reviewId"]].reset_index(drop=True)
    if query_id == 3:
        count = int((r["id"] == "taken_3") & (r["scoreSentiment"] == "POSITIVE")).sum() if False else int(((r["id"] == "taken_3") & (r["scoreSentiment"] == "POSITIVE")).sum())
        return pd.DataFrame({"positive_review_cnt": [count]})
    if query_id == 4:
        sub = r[r["id"] == "taken_3"]
        ratio = float((sub["scoreSentiment"] == "POSITIVE").sum()) / len(sub) if len(sub) > 0 else 0.0
        return pd.DataFrame({"positivity_ratio": [ratio]})
    if query_id in (5, 6, 7):
        movie_id = "ant_man_and_the_wasp_quantumania"
        sub = r[r["id"] == movie_id]
        merged = sub.merge(sub, on="id", suffixes=("_1", "_2"))
        merged = merged[merged["reviewId_1"] != merged["reviewId_2"]]
        if query_id == 5:
            merged = merged[merged["scoreSentiment_1"] == merged["scoreSentiment_2"]]
        else:
            merged = merged[merged["scoreSentiment_1"] != merged["scoreSentiment_2"]]
        return merged[["id", "reviewId_1", "reviewId_2"]].rename(columns={"reviewId_1": "reviewId1", "reviewId_2": "reviewId2"}).reset_index(drop=True)
    if query_id == 8:
        sub = r[r["id"] == "taken_3"]
        counts = sub.groupby("scoreSentiment").size().reset_index(name="count")
        return counts.rename(columns={"scoreSentiment": "scoreSentiment"})
    if query_id == 9:
        sub = r[r["id"] == "ant_man_and_the_wasp_quantumania"].copy()
        def parse_num(s):
            try:
                return float(str(s).split("/")[0])
            except (ValueError, AttributeError):
                return float("nan")
        sub["reviewScore"] = sub["originalScore"].apply(parse_num)
        return sub[["reviewId", "reviewScore"]].reset_index(drop=True)
    if query_id == 10:
        m = _load_movies()
        return m[["id", "audienceScore"]].rename(columns={"id": "movieId", "audienceScore": "movieScore"}).reset_index(drop=True)
    raise ValueError(f"Unknown query_id: {query_id}")


# ── Evaluation functions (aligned with SemBench MovieEvaluator) ───────────────

def evaluate_q1(sys_df: pd.DataFrame) -> QueryMetricRetrieval:
    """Retrieval with limit=5. sys_df must have column: reviewId."""
    return _retrieval_limit(sys_df, get_ground_truth(1), limit=5)


def evaluate_q2(sys_df: pd.DataFrame) -> QueryMetricRetrieval:
    """Retrieval with limit=5 for taken_3. sys_df must have column: reviewId."""
    return _retrieval_limit(sys_df, get_ground_truth(2), limit=5)


def evaluate_q3(sys_df: pd.DataFrame) -> QueryMetricAggregation:
    """Single-value aggregation (count). sys_df must have one row with a numeric column."""
    return _aggregation_single(sys_df, get_ground_truth(3))


def evaluate_q4(sys_df: pd.DataFrame) -> QueryMetricAggregation:
    """Single-value aggregation (ratio). sys_df must have one row with a numeric column."""
    return _aggregation_single(sys_df, get_ground_truth(4))


def evaluate_q5(sys_df: pd.DataFrame) -> QueryMetricRetrieval:
    """Pair retrieval with limit=10. sys_df must have columns: id, reviewId1, reviewId2."""
    return _review_pairs_limit(sys_df, get_ground_truth(5), limit=10)


def evaluate_q6(sys_df: pd.DataFrame) -> QueryMetricRetrieval:
    """Pair retrieval with limit=10. sys_df must have columns: id, reviewId1, reviewId2."""
    return _review_pairs_limit(sys_df, get_ground_truth(6), limit=10)


def evaluate_q7(sys_df: pd.DataFrame) -> QueryMetricRetrieval:
    """All-pairs retrieval (no limit). sys_df must have columns: id, reviewId1, reviewId2."""
    return _review_pairs(sys_df, get_ground_truth(7))


def evaluate_q8(sys_df: pd.DataFrame) -> QueryMetricAggregation:
    """Sentiment count aggregation. sys_df must have columns: scoreSentiment, count."""
    return _sentiment_counts(sys_df, get_ground_truth(8))


def evaluate_q9(sys_df: pd.DataFrame) -> QueryMetricRank:
    """Ranking by review score. sys_df must have columns: reviewId, reviewScore."""
    return _ranking(sys_df, get_ground_truth(9))


def evaluate_q10(sys_df: pd.DataFrame) -> QueryMetricRank:
    """Ranking by movie score. sys_df must have columns: movieId, movieScore."""
    return _ranking(sys_df, get_ground_truth(10))


def evaluate(query_id: int, sys_df: pd.DataFrame):
    """Convenience dispatcher: evaluate_q{query_id}(sys_df)."""
    fn = {
        1: evaluate_q1, 2: evaluate_q2, 3: evaluate_q3, 4: evaluate_q4,
        5: evaluate_q5, 6: evaluate_q6, 7: evaluate_q7, 8: evaluate_q8,
        9: evaluate_q9, 10: evaluate_q10,
    }[query_id]
    return fn(sys_df)


# ── Internal helpers (logic from SemBench) ────────────────────────────────────

def _retrieval_limit(sys_df, gt_df, limit):
    if len(sys_df) == 0:
        return QueryMetricRetrieval(precision=1.0 if len(gt_df) == 0 else 0.0)
    if len(gt_df) == 0:
        return QueryMetricRetrieval()

    sys_df = sys_df.head(limit)
    sys_col, gt_col = sys_df.columns[0], gt_df.columns[0]
    sys_ids = set(sys_df[sys_col].dropna().astype(str))
    gt_ids = set(gt_df[gt_col].dropna().astype(str))
    valid = sys_ids & gt_ids

    precision = len(valid) / len(sys_ids) if sys_ids else 0.0
    recall = min(len(valid), limit) / min(limit, len(gt_ids)) if gt_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return QueryMetricRetrieval(precision, recall, f1)


def _aggregation_single(sys_df, gt_df):
    m = QueryMetricAggregation()
    if len(sys_df) != 1 or len(gt_df) != 1:
        m.relative_error = 1.0
        m.absolute_error = float("inf")
        m.mean_absolute_percentage_error = 100.0
        return m

    def first_num(df):
        for c in df.columns:
            val = df[c].iloc[0]
            if isinstance(val, str):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
            if isinstance(val, (int, float)) and pd.notna(val):
                return float(val)
        return None

    sys_val, gt_val = first_num(sys_df), first_num(gt_df)
    if sys_val is None or gt_val is None:
        m.relative_error, m.absolute_error, m.mean_absolute_percentage_error = 1.0, float("inf"), 100.0
        return m
    m.absolute_error = abs(sys_val - gt_val)
    if gt_val != 0:
        m.relative_error = m.absolute_error / abs(gt_val)
        m.mean_absolute_percentage_error = m.relative_error * 100
    else:
        m.relative_error = float("inf") if sys_val != 0 else 0.0
        m.mean_absolute_percentage_error = m.relative_error * 100
    return m


def _make_pair_set(df):
    """Convert DataFrame (id, reviewId1, reviewId2) to set of (movie_id, sorted_tuple)."""
    if len(df.columns) < 3:
        return set()
    cols = list(df.columns)
    result = set()
    for _, row in df.iterrows():
        mid, v1, v2 = row[cols[0]], row[cols[1]], row[cols[2]]
        if pd.notna(mid) and pd.notna(v1) and pd.notna(v2):
            result.add((mid, tuple(sorted([str(v1), str(v2)]))))
    return result


def _review_pairs(sys_df, gt_df):
    if len(sys_df) == 0:
        return QueryMetricRetrieval(precision=1.0 if len(gt_df) == 0 else 0.0)
    if len(gt_df) == 0:
        return QueryMetricRetrieval()
    if len(sys_df.columns) < 3 or len(gt_df.columns) < 3:
        return QueryMetricRetrieval()

    sys_pairs = _make_pair_set(sys_df)
    gt_pairs = _make_pair_set(gt_df)
    correct = sys_pairs & gt_pairs
    precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
    recall = len(correct) / len(gt_pairs) if gt_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return QueryMetricRetrieval(precision, recall, f1)


def _review_pairs_limit(sys_df, gt_df, limit):
    if len(sys_df) == 0:
        return QueryMetricRetrieval(precision=1.0 if len(gt_df) == 0 else 0.0)
    if len(gt_df) == 0:
        return QueryMetricRetrieval()
    if len(sys_df.columns) < 3 or len(gt_df.columns) < 3:
        return QueryMetricRetrieval()

    sys_df = sys_df.head(limit)
    sys_pairs = _make_pair_set(sys_df)
    gt_pairs = _make_pair_set(gt_df)
    correct = sys_pairs & gt_pairs
    precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
    recall = min(len(correct), limit) / min(limit, len(gt_pairs)) if gt_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return QueryMetricRetrieval(precision, recall, f1)


def _sentiment_counts(sys_df, gt_df):
    if len(sys_df) == 0 or len(gt_df) == 0:
        return QueryMetricAggregation(relative_error=1.0, absolute_error=float("inf"), mean_absolute_percentage_error=100.0)
    if len(sys_df.columns) < 2 or len(gt_df.columns) < 2:
        return QueryMetricAggregation(relative_error=1.0, absolute_error=float("inf"), mean_absolute_percentage_error=100.0)

    def parse_counts(df):
        out = {}
        scol, ccol = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            s, c = row[scol], row[ccol]
            if pd.notna(s) and pd.notna(c):
                try:
                    out[str(s).strip().upper()] = float(c)
                except (ValueError, TypeError):
                    pass
        return out

    sys_counts = parse_counts(sys_df)
    gt_counts = parse_counts(gt_df)
    if not sys_counts or not gt_counts:
        return QueryMetricAggregation(relative_error=1.0, absolute_error=float("inf"), mean_absolute_percentage_error=100.0)

    total_abs, total_rel, valid = 0.0, 0.0, 0
    for sentiment in set(sys_counts) | set(gt_counts):
        sc, gc = sys_counts.get(sentiment, 0.0), gt_counts.get(sentiment, 0.0)
        total_abs += abs(sc - gc)
        if gc != 0:
            total_rel += abs(sc - gc) / abs(gc)
            valid += 1
        elif sc != 0:
            total_rel += 1.0
            valid += 1

    rel = total_rel / valid if valid > 0 else 0.0
    return QueryMetricAggregation(relative_error=rel, absolute_error=total_abs, mean_absolute_percentage_error=rel * 100)


def _ranking(sys_df, gt_df):
    from scipy.stats import spearmanr, kendalltau

    if len(sys_df) == 0 or len(gt_df) == 0:
        return QueryMetricRank()
    if len(sys_df.columns) < 2 or len(gt_df.columns) < 2:
        return QueryMetricRank()

    def to_score_map(df):
        id_col, score_col = df.columns[0], df.columns[1]
        out = {}
        for _, row in df.iterrows():
            k, v = row[id_col], row[score_col]
            if pd.notna(k) and pd.notna(v):
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    pass
        return out

    sys_scores = to_score_map(sys_df)
    gt_scores = to_score_map(gt_df)
    common = set(sys_scores) & set(gt_scores)
    if len(common) < 2:
        return QueryMetricRank()

    sv = [sys_scores[k] for k in common]
    gv = [gt_scores[k] for k in common]

    try:
        sp = spearmanr(sv, gv).correlation
        sp = 0.0 if pd.isna(sp) else sp
    except Exception:
        sp = 0.0
    try:
        kt = kendalltau(sv, gv).correlation
        kt = 0.0 if pd.isna(kt) else kt
    except Exception:
        kt = 0.0

    return QueryMetricRank(spearman_correlation=sp, kendall_tau=kt)
