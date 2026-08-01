"""Benchmark predicate-aware HNSW filter representations (paper Table 5)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import tuple_row

from ours.utils import make_bitmap, quote, resolve_predicate_ids

BLOOM_MAGIC = b"HBLM"
UINT64_MASK = (1 << 64) - 1


def murmurhash64(value: int) -> int:
    value &= UINT64_MASK
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & UINT64_MASK
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & UINT64_MASK
    value ^= value >> 33
    return value & UINT64_MASK


def bloom_positions(
    real_id: int, seed_one: int, seed_two: int, bit_count: int, hash_count: int
) -> Iterable[int]:
    first = murmurhash64(real_id ^ seed_one) % bit_count
    second = murmurhash64(real_id ^ seed_two) % bit_count or 1
    for index in range(hash_count):
        yield (first + index * second) % bit_count


def next_prime(value: int) -> int:
    candidate = max(2, value)
    while True:
        if all(candidate % divisor for divisor in range(2, int(math.sqrt(candidate)) + 1)):
            return candidate
        candidate += 1


def make_bloom(valid_ids: Iterable[int], bits_per_element: int, seed: int) -> bytes:
    ids = [int(real_id) for real_id in valid_ids if real_id is not None and real_id >= 0]
    bit_count = next_prime(max(1024, len(ids) * bits_per_element))
    hash_count = max(1, round(bits_per_element * math.log(2)))
    rng = random.Random(seed)
    seed_one = rng.getrandbits(64)
    seed_two = rng.getrandbits(64)
    bitset = bytearray(bit_count // 8 + 1)
    for real_id in ids:
        for position in bloom_positions(real_id, seed_one, seed_two, bit_count, hash_count):
            bitset[position >> 3] |= 1 << (position & 7)
    return struct.pack("<4sIIQQ", BLOOM_MAGIC, hash_count, bit_count, seed_one, seed_two) + bitset


def bloom_contains(payload: bytes, real_id: int) -> bool:
    magic, hash_count, bit_count, seed_one, seed_two = struct.unpack("<4sIIQQ", payload[:28])
    if magic != BLOOM_MAGIC:
        raise ValueError("invalid Bloom filter header")
    bitset = payload[28:]
    return all(
        bitset[position >> 3] & (1 << (position & 7))
        for position in bloom_positions(real_id, seed_one, seed_two, bit_count, hash_count)
    )


def build_where(predicates: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    clauses = []
    params: dict[str, Any] = {}
    for index, predicate in enumerate(predicates):
        key = f"p{index}"
        operator = predicate["operator"]
        if operator == "in":
            clauses.append(f'{quote(predicate["attribute"])} = ANY(%({key})s)')
            params[key] = list(predicate["value"])
        else:
            clauses.append(f'{quote(predicate["attribute"])} {operator} %({key})s')
            params[key] = predicate["value"]
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def exact_query(
    conn: psycopg.Connection,
    table: str,
    id_column: str,
    vector_column: str,
    query_vector: Vector,
    predicates: list[dict[str, Any]],
    result_count: int,
) -> list[int]:
    where, params = build_where(predicates)
    params.update({"query": query_vector, "limit": result_count})
    statement = (
        f'SELECT {quote(id_column)} FROM {quote(table)} WHERE {where} '
        f'ORDER BY {quote(vector_column)} <-> %(query)s LIMIT %(limit)s'
    )
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_indexonlyscan = off")
        cur.execute("SET LOCAL enable_seqscan = on")
        cur.execute(statement, params)
        return [int(row[0]) for row in cur.fetchall()]


def filtered_query(
    conn: psycopg.Connection,
    table: str,
    id_column: str,
    vector_column: str,
    query_vector: Vector,
    filter_payload: bytes,
    valid_ids: set[int],
    *,
    result_count: int,
    candidate_count: int,
    ef_search: int,
    enable_two_hop: bool,
    bloom: bool,
) -> list[int]:
    statement = (
        f'SELECT {quote(id_column)} FROM {quote(table)} '
        f'WHERE {quote(vector_column)} OPERATOR(public.<->#) %(filter)s::bytea '
        f'ORDER BY {quote(vector_column)} <-> %(query)s LIMIT %(limit)s'
    )
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {min(max(ef_search, candidate_count), 1000)}")
        cur.execute(f"SET LOCAL hnsw.enable_2hop = {'on' if enable_two_hop else 'off'}")
        cur.execute(
            "SELECT set_config('hnsw.id_map_table', %s, true)",
            (f"public.{table}_id_map",),
        )
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute(
            statement,
            {"filter": psycopg.Binary(filter_payload), "query": query_vector, "limit": candidate_count},
        )
        ids = [int(row[0]) for row in cur.fetchall()]
    if bloom:
        ids = [real_id for real_id in ids if real_id in valid_ids]
    return ids[:result_count]


def recall(expected: set[int], actual: Iterable[int]) -> float:
    return len(expected & set(actual)) / len(expected) if expected else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url", default=os.environ.get("IMDB_DATABASE_URL", "postgresql://localhost/imdb")
    )
    parser.add_argument("--workload", type=Path, default=Path("imdb_data/workload/w2/w2_queries_100.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("imdb_data/results/w2/w2_queries_100_gt.json"))
    parser.add_argument("--table", default="imdb_t1_hnsw")
    parser.add_argument("--logical-table", default="imdb_t1")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--vector-column", default=None)
    parser.add_argument("--ef-search", type=int, default=400)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--adaptive-threshold", type=float, default=1 / 32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/filtered_search.json"))
    args = parser.parse_args(argv)

    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    queries = workload["queries"][: args.limit] if args.limit else workload["queries"]
    truth = {
        row["query_id"]: {
            int(answer["id"] if isinstance(answer, dict) else answer[0])
            for answer in row["answer"]
        }
        for row in json.loads(args.ground_truth.read_text(encoding="utf-8"))["results"]
    }
    arms = ["exact", "bitmap_2hop_on", "bitmap_2hop_off", "bitmap_adaptive", "bloom_4", "bloom_12", "bloom_20"]
    totals = {arm: {"seconds": 0.0, "recall": 0.0} for arm in arms}

    conn = psycopg.connect(args.database_url)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), max({quote(args.id_column)}) FROM {quote(args.table)}")
        table_count, maximum_id = map(int, cur.fetchone())
    for query_index, query in enumerate(queries):
        signal = query["scoring"]["signals"][0]
        vector_column = args.vector_column or signal["field"]
        query_vector = Vector(signal["query_embed"])
        result_count = int(query["K"])
        predicates = [
            predicate for predicate in query.get("predicates", [])
            if predicate["table"].lower() == args.logical_table.lower()
        ]
        valid_ids = resolve_predicate_ids(conn, args.table, predicates, id_col=args.id_column)

        started = time.perf_counter()
        exact = exact_query(
            conn, args.table, args.id_column, vector_column,
            query_vector, predicates, result_count,
        )
        totals["exact"]["seconds"] += time.perf_counter() - started
        totals["exact"]["recall"] += recall(truth[query["query_id"]], exact)
        conn.commit()

        bitmap = make_bitmap(valid_ids, maximum_id)
        configurations = [
            ("bitmap_2hop_on", bitmap, True, False),
            ("bitmap_2hop_off", bitmap, False, False),
            ("bitmap_adaptive", bitmap, len(valid_ids) / table_count <= args.adaptive_threshold, False),
            ("bloom_4", make_bloom(valid_ids, 4, query_index), False, True),
            ("bloom_12", make_bloom(valid_ids, 12, query_index), False, True),
            ("bloom_20", make_bloom(valid_ids, 20, query_index), False, True),
        ]
        for arm, payload, two_hop, is_bloom in configurations:
            started = time.perf_counter()
            result = filtered_query(
                conn, args.table, args.id_column, vector_column, query_vector,
                payload, valid_ids, result_count=result_count,
                candidate_count=args.candidate_count, ef_search=args.ef_search,
                enable_two_hop=two_hop, bloom=is_bloom,
            )
            totals[arm]["seconds"] += time.perf_counter() - started
            totals[arm]["recall"] += recall(truth[query["query_id"]], result)
            conn.commit()
    conn.close()

    count = len(queries)
    results = {
        arm: {
            "recall": values["recall"] / count,
            "qps": count / values["seconds"],
            "seconds": values["seconds"],
        }
        for arm, values in totals.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())