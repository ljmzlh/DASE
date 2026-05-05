# Movie Cascade — Final Summary

Cascade results across Q1–Q10 of the Movie scenario. Latency is **re-estimated under paper-day BQ parallelism** (per-Q measured where possible, workload-proxy fallback for queries whose baseline was aborted).

Cost = sum of (calibrated per-row × #LLM calls); for cascade this is BQ Stage 2 only.

## Cascade core table

| Q | op | score | cost ($) | latency (s) |
|---|---|---:|---:|---:|
| Q1 | F+L | 1.00 | 0.000127 | 7.62 |
| Q2 | F+L | 1.00 | 0.000136 | 6.88 |
| Q3 | F | 0.82 | 0.000651 | 8.79 |
| Q4 | F | 0.82 | 0.000651 | 9.06 |
| Q5 | J+L | 0.82 | 0.000618 | 9.74 |
| Q6 | J+L | 0.95 | 0.000655 | 9.53 |
| Q7 | J | 0.72 | 0.000950 | 6.29 |
| Q8 | C | 0.89 | 0.000651 | 8.72 |
| Q9 | R | 0.75 | 0.006027 | 3.94 |
| Q10 | R | 0.41 | 0.084563 | 20.51 |

## Comparison with paper BQ baseline and paper DASE

| Q | op | paper BQ score / cost / lat | paper DASE score | cascade score / cost / lat | cost ↓ | lat ↓ | parallelism (used) |
|---|---|---|---:|---|---:|---:|---:|
| Q1 | F+L | 1.00 / $0.05 / 26.3s | 1.00 | 1.00 / $0.0001267 / 7.62s | 395× | 3.5× | 2.01× (measured) |
| Q2 | F+L | 1.00 / $0.003 / 9.5s | 1.00 | 1.00 / $0.0001357 / 6.88s | 22× | 1.4× | 1.35× (measured) |
| Q3 | F | 0.64 / $0.003 / 11.0s | 0.82 | 0.82 / $0.0006514 / 8.79s | 5× | 1.3× | 1.49× (measured) |
| Q4 | F | 0.64 / $0.003 / 11.4s | 0.82 | 0.82 / $0.0006514 / 9.06s | 5× | 1.3× | 1.44× (measured) |
| Q5 | J+L | 0.89 / $1.01 / 54.5s | 1.00 | 0.82 / $0.0006182 / 9.74s | 1634× | 5.6× | 2.01× (Q1_proxy) |
| Q6 | J+L | 0.69 / $1 / 54.5s | 1.00 | 0.95 / $0.0006552 / 9.53s | 1526× | 5.7× | 2.01× (Q1_proxy) |
| Q7 | J | 0.70 / $3.31 / 198.3s | 0.73 | 0.72 / $0.0009495 / 6.29s | 3486× | 31.5× | 62.66× (Q10_proxy) |
| Q8 | C | 0.76 / $0.003 / 10.9s | 0.89 | 0.89 / $0.0006514 / 8.72s | 5× | 1.2× | 1.50× (measured) |
| Q9 | R | 0.78 / $0.02 / 13.3s | 0.66 | 0.75 / $0.006027 / 3.94s | 3× | 3.4× | 62.66× (Q10_proxy) |
| Q10 | R | 0.44 / $0.13 / 32.1s | 0.43 | 0.41 / $0.08456 / 20.51s | 2× | 1.6× | 62.66× (measured) |

## Notes

- **Latency** is re-estimated under per-Q paper BQ parallelism: `cascade_BQ_wall = cascade_slot_ms × paper_wall / our_bq_slot_ms; cascade_total = cascade_BQ_wall + dase_wall`. This adjusts away the BQ slot-allocation variance we suffered today.
- For Qs whose baseline was aborted (Q5, Q6, Q7, Q9), the paper parallelism is taken from a workload-similar Q: Q5/Q6 → Q1 (F+L LIMIT short-circuit), Q7 → Q10 (no LIMIT, large throughput), Q9 → Q10 (same AI.SCORE operator).
- **Score**: F1 for retrieval/join queries (Q1, Q2, Q5–Q7); 1/(1+rel_err) aggregation score for Q3/Q4/Q8; Spearman correlation for Q9/Q10.
- **Cost**: sum of LLM-call cost for cascade BQ Stage 2 only (Stage 1 CTAS has no LLM cost; dase embedding cost is amortized across all queries).
