# -*- coding: utf-8 -*-
'''
usage:
python w4_ours.py --ti_table w4_ti_0.35 --mode exp --filter_type bitmap \
     --twohop_strategy disable

python w4_ours.py --ti_table w4_ti_0.35 --mode exp --filter_type bloom --bloom_b 8 \
    --twohop_strategy disable


'''
import os
import argparse
import random
import struct
from collections import deque
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg import sql
from tqdm import tqdm
from utils import output_plan
import time
from eval import evaluate

# optimal hash num for b
b_k_mapping = {
    1: 1, 2: 1, 3: 2, 4: 3, 5: 3,
    6: 4, 7: 5, 8: 6, 9: 6, 10: 7,
    11: 8, 12: 8, 13: 9, 14: 10, 15: 10,
    16: 11, 17: 12, 18: 12, 19: 13, 20: 14,
}
HNSW_BLOOM_MAGIC = b"HBLM"
UINT64_MASK = (1 << 64) - 1



# 使用 psycopg3 的 notice handler 收集更多 notice
NOTICE_BUFFER_SIZE = 5000
notice_buffer = deque(maxlen=NOTICE_BUFFER_SIZE)
def _format_notice(diag):
    # psycopg3 notice Diagnostic 对象，提取主要字段
    sev = diag.severity_nonlocalized or diag.severity
    msg = diag.message_primary
    sqlstate = diag.sqlstate
    detail = diag.message_detail
    hint = diag.message_hint
    ctx = diag.context
    lines = []
    head = " | ".join([p for p in [sev, sqlstate, msg] if p])
    if head:
        lines.append(head)
    if detail:
        lines.append(f"detail: {detail}")
    if hint:
        lines.append(f"hint: {hint}")
    if ctx:
        lines.append(f"context: {ctx}")
    return "\n".join(lines) if lines else str(diag)

def _notice_handler(diag, buf=notice_buffer):
    buf.append(_format_notice(diag))



def murmurhash64_py(value: int) -> int:
    """软件实现 murmurhash64，与 C 端保持一致。"""
    value &= UINT64_MASK
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & UINT64_MASK
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & UINT64_MASK
    value ^= value >> 33
    return value & UINT64_MASK


def _bloom_positions(realid: int, seed1: int, seed2: int, bit_size: int, k: int):
    """double hashing 生成 k 个 bit 位置。"""
    h1 = murmurhash64_py(realid ^ seed1) % bit_size
    h2 = murmurhash64_py(realid ^ seed2) % bit_size
    if h2 == 0:
        h2 = 1
    for i in range(k):
        yield (h1 + i * h2) % bit_size

def make_bitmap_filter(selected_ids: Iterable[int], max_id: int) -> bytes:
    bitset = bytearray(max_id // 8 + 1)
    for aid in selected_ids:
        if aid is None or aid < 0 or aid > max_id:
            continue
        bitset[aid >> 3] |= 1 << (aid & 0x07)
    return bytes(bitset)


def make_bloom_header(bloom_k: int, bit_size: int, seed1: int, seed2: int) -> bytes:
    return struct.pack(
        "<4sIIQQ",
        HNSW_BLOOM_MAGIC,
        bloom_k,
        bit_size,
        seed1 & UINT64_MASK,
        seed2 & UINT64_MASK,
    )

def parse_bloom_filter(filter):
    # 接受实现 buffer 协议的对象（bytes/bytearray/memoryview/psycopg.Binary 等）
    try:
        data = memoryview(filter).tobytes()
    except Exception as e:
        raise TypeError(f"Unsupported filter type: {type(filter)}") from e
    if len(data) < 28:  # 4s + I + I + Q + Q
        raise ValueError(f"Bloom filter too short: {len(data)} bytes, type={type(filter)}")
    magic, bloom_k, bit_size, seed1, seed2 = struct.unpack("<4sIIQQ", data[:28])
    if magic != HNSW_BLOOM_MAGIC:
        raise ValueError(f"Invalid bloom filter magic: {magic}")
    bitmap = data[28:]
    return bloom_k, bit_size, seed1, seed2, bitmap


def make_bloom_filter(selected_ids: Iterable[int], max_id, args) -> bytes:
    def find_next_prime(n):
        while 1:
            bo = True
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    bo = False
                    break
            if bo:
                return n
            n += 1
    
    ids = [aid for aid in selected_ids if aid is not None and aid >= 0]
    
    bit_size = max(1024, len(ids) * args.bloom_b)
    bit_size = find_next_prime(bit_size)
 
    bytea = bytearray(bit_size // 8 + 1)
    seed1 = random.getrandbits(64)
    seed2 = random.getrandbits(64)

    # 为每个 id 单独生成 bloom 位置
    for aid in ids:
        for pos in _bloom_positions(aid, seed1, seed2, bit_size, args.bloom_k):
            bytea[pos >> 3] |= 1 << (pos & 0x07)

    header = make_bloom_header(args.bloom_k, bit_size, seed1, seed2)
    return header + bytea


def make_filter(selected_ids: Iterable[int], max_id: int, args) -> bytes:
    if args.filter_type == "bitmap":
        return make_bitmap_filter(selected_ids, max_id)
    if args.filter_type == "bloom":
        return make_bloom_filter(selected_ids, max_id, args)
    raise ValueError(f"unknown filter_type: {args.filter_type}")


def probe_bloom_filter(filter, qid, args):
    bloom_k, bit_size, seed1, seed2, bitmap = parse_bloom_filter(filter)
    pos = _bloom_positions(qid, seed1, seed2, bit_size, bloom_k)
    for pos in pos:
        if bitmap[pos >> 3] & (1 << (pos & 0x07)) == 0:
            return False
    return True
    


def main(args: argparse.Namespace):
    # 固定随机种子，保证 Bloom Filter 在相同输入下可复现
    random.seed(20241222)

    ti_table = args.ti_table
    threshold = args.ti_table.split('_')[-1]
    if(args.filter_type == "bloom"):
        args.bloom_k = b_k_mapping[args.bloom_b]
    
    DST = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database="molecule", user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", ""))
    conn=psycopg.connect(**DST)
    conn.add_notice_handler(_notice_handler)
    cur = conn.cursor()
    
    sels = [] 

    qn = 1000 if args.mode == "exp" else 1000
    print(f"{args.mode} on qn={qn}")

    if(args.mode == 'exp'):
        method_name = args.filter_type
        if(args.filter_type == "bloom"):
            method_name += f"{args.bloom_b}"
        method_name += f"_{args.twohop_strategy}_2hop"
        output_fn = f"w4_result/join_index({method_name})_results_{ti_table}.csv"
        outf=open(output_fn, "w")
        outf.write('id,rank,a1_id,a2_id,paper_id,summary_distance\n')

        time_outf = open(f"w4_result/time_{method_name}_results_{ti_table}.csv", "w")
        time_outf.write('id,query_time,select_id2_time,select_paper_time,make_filter_time,psql_time\n')


    cur.execute("SET LOCAL enable_bitmapscan = on;")
    cur.execute("SET LOCAL enable_seqscan = off;")
    cur.execute("SET LOCAL enable_indexscan = on;")

    # 确保过滤查询能拿到 tid map；SET 之后底层会按需自动加载
    cur.execute("SET hnsw.id_map_table = 'public.w4_corpus_id_map';")

    st = time.time()

    
    select_id2_time = 0
    select_paper_time = 0
    make_filter_time = 0
    psql_time =0
    for id in tqdm(range(1,qn+1)):
        
        cur.execute("SELECT * FROM w4_workload WHERE id = %s", (id,))
        row = cur.fetchone()
        if row is not None:
            id, q_topic, q_field, construct_a1_id, construct_a2_id, construct_dism, q_topic_gem_1536 = row
        else:
            print(f"No record found for id={id}")
            continue
        
        # construct filter_map, a1.field = q_field
        start_time = time.time()
        query_start_time = time.time()
        max_id = 53732
        # 只用查一边，因为ti是对称的
        cur.execute(
            sql.SQL("SELECT id1, id2 FROM {ti_table} WHERE field1 = %s").format(
                ti_table=sql.Identifier(ti_table)
            ),
            (q_field,)
        )

        rows = cur.fetchall()
        id2_to_id1 = {}

        for row in rows:
            id2_to_id1[row[1]] = row[0]


        this_select_id2_time = time.time() - start_time
        select_id2_time += this_select_id2_time

        
        if(args.twohop_strategy == "enable"):
            cur.execute("SET LOCAL hnsw.enable_2hop = true;")
        elif(args.twohop_strategy == "disable"):
            cur.execute("SET LOCAL hnsw.enable_2hop = false;")
        elif(args.twohop_strategy == "adaptive"):
            selectivity = len(id2_to_id1) / max_id
            if(selectivity > 0.03125): # 1/32
                cur.execute("SET LOCAL hnsw.enable_2hop = false;")
            else:
                cur.execute("SET LOCAL hnsw.enable_2hop = true;")
        

        if(args.mode == 'profile'):
            sels.append((id, len(id2_to_id1) / max_id))
            continue
        elif(args.mode == 'debug'):
            print(f"distinct a2.id: {len(id2_to_id1)}")
            print()
            sb=input('press enter to continue')
        
        
        start_time = time.time()

        filter_map = make_filter(id2_to_id1.keys(), max_id, args)

        this_make_filter_time = time.time() - start_time
        make_filter_time += this_make_filter_time
        start_time = time.time()

        base_sql =f"""
            SELECT id AS neighbor_id,
                (summary_gem_1536 <-> %s::vector) AS dist
            FROM w4_corpus
            WHERE (summary_gem_1536 OPERATOR(public.<->#) %s::bytea)
            ORDER BY summary_gem_1536 <-> %s::vector ASC
            LIMIT {args.hardness};
        """

        bottom_sql = f"""
            SELECT id,(summary_gem_1536 <-> %s::vector)
            FROM w4_corpus
            ORDER BY summary_gem_1536 <-> %s::vector ASC
            LIMIT {args.hardness};
        """

        if(args.mode == 'bottom'):
            cur.execute(bottom_sql, (q_topic_gem_1536, q_topic_gem_1536))
            results = cur.fetchall()
        if(args.mode == 'exp'):
            cur.execute(base_sql, (q_topic_gem_1536, psycopg.Binary(filter_map), q_topic_gem_1536))
            results = cur.fetchall()
            this_psql_time = time.time() - start_time
            psql_time += this_psql_time

            

            start_time = time.time()
            candidate_id2s =  [row[0] for row in results]
            cur.execute(
                "SELECT id, paper_id FROM w4_corpus WHERE id = ANY(%s)",
                (candidate_id2s,),
            )
            rows = cur.fetchall()
            id2_to_paper_id = {row[0]: row[1] for row in rows}

            this_select_paper_time = time.time() - start_time
            select_paper_time += this_select_paper_time


            rank = 1
            seen_paper_id = set()

            for row in results:
                a2_id, summary_distance = row

                # if bloom, check false positive
                if(args.filter_type == "bloom" and a2_id not in id2_to_id1):
                    continue

                # save results to csv
                a1_id = id2_to_id1[a2_id]
                paper_id = id2_to_paper_id[a2_id]
                if paper_id in seen_paper_id:
                    continue

                seen_paper_id.add(paper_id)
                outf.write(f'{id},{rank},{a1_id},{a2_id},{paper_id},{summary_distance}\n')
                rank += 1
                if(rank > 10):
                    break

            time_outf.write(f'{id},{time.time() - query_start_time},{this_select_id2_time},{this_select_paper_time},{this_make_filter_time},{time.time() - start_time}\n')
            
            outf.flush()
            time_outf.flush()

        elif(args.mode == 'debug'):
            debug_sql = 'EXPLAIN ' + base_sql

            cur.execute(debug_sql, (q_topic_gem_1536, psycopg.Binary(filter_map), q_topic_gem_1536))
            results = cur.fetchall()

            for row in results:
                line = row[0]
                output_plan(line)
            sb=input("press enter to continue")

            cur.execute(base_sql, (q_topic_gem_1536, psycopg.Binary(filter_map), q_topic_gem_1536))
            results = cur.fetchall()

            print(f"\n=== pgvector elog notices for id={id} ===\n")
            if notice_buffer:
                print("\n".join(notice_buffer))
                # 如需避免重复打印，打印后清空
                notice_buffer.clear()
            print("=" * 50)
            
            sb=input('press enter to continue')




            print('result: a1.id, a2.id, summary_distance')
            for row in results:
                a2_id, summary_distance = row
                try:
                    a1_id = id2_to_id1[a2_id]
                except KeyError:
                    a1_id = None
                print(a1_id, a2_id, summary_distance)
        
        




    if(args.mode =='profile'):
            plt.hist([sel for id, sel in sels], bins=100)
            plt.xlabel('Selectivity')
            plt.ylabel('Frequency')
            plt.title('Selectivity Distribution')
            plt.show()
            plt.savefig('selectivity_distribution.png')
            print('avg selectivity: ', sum([sel for id, sel in sels]) / len(sels))
            print('max selectivity: ', max([sel for id, sel in sels]))
            print('min selectivity: ', min([sel for id, sel in sels]))
            print('std selectivity: ', np.std([sel for id, sel in sels]))

            # print quatile of selectivity
            print('25% quantile of selectivity: ', np.percentile([sel for id, sel in sels], 25))
            print('50% quantile of selectivity: ', np.percentile([sel for id, sel in sels], 50))
            print('75% quantile of selectivity: ', np.percentile([sel for id, sel in sels], 75))

            # save selectivity to csv
            with open(f'w4_result/selectivity_{threshold}.csv', 'w') as f:
                f.write('id,selectivity\n')
                for id, sel in sels:
                    f.write(f'{id},{sel}\n')


    time_cost = time.time() - st
    qps = qn / time_cost
    recall = None
    if(args.mode == 'exp'):
        '''
        --gt w4_result/gt_results_0.35.csv \
    --eval 'w4_result/join_index(bloom20_disable_2hop)_results_w4_ti_0.35.csv' \
    --selectivity w4_result/selectivity_0.35.csv
        '''
        gt_fn = f"w4_result/gt_results_{threshold}.csv"
        selectivity_fn = f"w4_result/selectivity_{threshold}.csv"
        _, recall = evaluate(gt_fn, output_fn, selectivity_fn)

    print('total time: ', time_cost)
    print('select id2 time: ', select_id2_time)
    print('select paper time: ', select_paper_time)
    print('make filter time: ', make_filter_time)
    print('psql time: ', psql_time)
    
    return qps, recall



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ti_table", type=str, help="指定ti表名")
    parser.add_argument("--mode", type=str, default="exp", help="exp/debug/profile")
    parser.add_argument("--filter_type", type=str, default="bitmap", choices=["bitmap", "bloom"])
    parser.add_argument("--bloom_b", type=int, default=6, help= "bits per element")
    parser.add_argument("--twohop_strategy", type=str, default="disable", choices=["disable", "enable", "adaptive"])
    parser.add_argument("--hardness", type=int, default=20, help="hardness of the search (num. of candidates from the index to output top10)")
    args = parser.parse_args()

    qps, recall = main(args)
    print(f"QPS: {qps}, recall: {recall}")