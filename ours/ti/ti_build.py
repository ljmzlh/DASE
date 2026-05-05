# build ti (threshold index) with pgvector hnsw query

'''
usage (from /dase/):
python -m ours.ti.ti_build --config imdb    --method directhnsw_range --metric l2 --threshold 0.6
python -m ours.ti.ti_build --config w4      --method directhnsw_range --metric l2 --threshold 0.6
python -m ours.ti.ti_build --config w5_molecule --method directhnsw_range --metric l2 --threshold 0.7

Per-build configs (table_left/right, id_col, vec_col, cols) live in ti_config.json.
CLI flags override config defaults.
'''

import json
import psycopg2
from .ti_build_utils import laterCrossJoin_HNSWTopK, loop_HNSWTopk, direct_HNSWTopk
from .ti_build_utils import direct_RangeSearch
from .ti_build_utils import loop_HNSWRange, laterCrossJoin_HNSWRange, direct_HNSWRange
from .ti_materialize import materialize_ti
import os
import argparse


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_configs():
    with open(os.path.join(SCRIPT_DIR, "ti_config.json")) as f:
        return json.load(f)


def main():
    CONFIGS = load_configs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        choices=list(CONFIGS.keys()),
                        help=f"Build config name: {list(CONFIGS.keys())}")
    parser.add_argument("--method", type=str, required=True,
                        choices=["lateralhnsw_topk", "loophnsw_topk", "direct_topk",
                                 "direct_range", "loophnsw_range", "lateralhnsw_range",
                                 "directhnsw_range"])
    parser.add_argument("--metric", type=str, required=True, help='l2/ip')
    parser.add_argument("--threshold", type=float, required=True)
    # optional overrides
    parser.add_argument("--vec_col", type=str, default=None,
                        help="override vec_col on both sides (ignored if --vec_col_left/right given)")
    parser.add_argument("--vec_col_left", type=str, default=None)
    parser.add_argument("--vec_col_right", type=str, default=None)
    parser.add_argument("--ef_search", type=int, default=None,
                        help="HNSW ef_search; overrides config's ef_search if given")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--source_batch", type=int, default=2048)
    args = parser.parse_args()

    # --- 从 config 加载配置，CLI 可覆盖 ---
    wl = CONFIGS[args.config]

    DSN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), database=wl["db"],
               user=os.environ.get("PGUSER", "postgres"),
               password=os.environ.get("PGPASSWORD", ""))
    TABLE_LEFT = wl["table_left"]
    TABLE_RIGHT = wl["table_right"]
    # vec_col / id_col: 优先 left/right 分开；否则回退到 workload 里的单一 vec_col / id_col
    VEC_COL_LEFT = args.vec_col_left or wl.get("vec_col_left") or args.vec_col or wl.get("vec_col")
    VEC_COL_RIGHT = args.vec_col_right or wl.get("vec_col_right") or args.vec_col or wl.get("vec_col")
    assert VEC_COL_LEFT and VEC_COL_RIGHT, "workload or CLI must define vec_col (or vec_col_left + vec_col_right)"
    ID_COL_LEFT = wl.get("id_col_left", wl.get("id_col"))
    ID_COL_RIGHT = wl.get("id_col_right", wl.get("id_col"))
    assert ID_COL_LEFT and ID_COL_RIGHT, "workload must define id_col or id_col_left + id_col_right"
    SOURCE_BATCH = args.source_batch
    TOPK = args.topk
    EF_SEARCH = args.ef_search if args.ef_search is not None else wl.get("ef_search")
    assert EF_SEARCH is not None, "ef_search must be set via --ef_search or config's ef_search"
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ti_cache", args.config)

    if args.metric == 'l2':
        TI_L2_THRESHOLD = args.threshold
        TI_IP_THRESHOLD = 1/2 * (TI_L2_THRESHOLD**2) - 1
    else:
        raise ValueError('not support')

    # 建立数据库连接
    conn = psycopg2.connect(**DSN)
    conn.autocommit = False

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    threshold = TI_L2_THRESHOLD if args.metric == 'l2' else TI_IP_THRESHOLD
    output_file = f"{OUTPUT_DIR}/ti_{TABLE_LEFT}_{TABLE_RIGHT}_{args.method}_{args.metric}_{threshold}.tsv"

    # 调用函数，传入所需参数（两表形式，TABLE_LEFT==TABLE_RIGHT 时为 self-join）
    # 其他 method 仍只接受单一 vec_col；异构 vec_col 仅 directhnsw_range 支持。
    # 若用户同时给了 vec_col_left/right 且二者不同，非 directhnsw_range 的 method 会 assert。
    if args.method != "directhnsw_range" and VEC_COL_LEFT != VEC_COL_RIGHT:
        raise ValueError(
            f"method={args.method} only supports a single vec_col, but got "
            f"vec_col_left={VEC_COL_LEFT!r} != vec_col_right={VEC_COL_RIGHT!r}. "
            "Use --method directhnsw_range for heterogeneous vec_col."
        )
    VEC_COL = VEC_COL_LEFT

    if args.method == "lateralhnsw_topk":
        laterCrossJoin_HNSWTopK(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col=VEC_COL,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=TOPK,
            ef_search=EF_SEARCH,
            ti_ip_threshold=TI_IP_THRESHOLD,
            output_file=output_file,
        )
    elif args.method == "loophnsw_topk":
        loop_HNSWTopk(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col=VEC_COL,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=TOPK,
            ef_search=EF_SEARCH,
            ti_threshold=threshold,
            output_file=output_file,
            metric=args.metric,
        )
    elif args.method == "direct_topk":
        direct_HNSWTopk(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col=VEC_COL,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=TOPK,
            ef_search=EF_SEARCH,
            ti_threshold=threshold,
            output_file=output_file,
            metric=args.metric,
        )
    elif args.method == "direct_range":
        direct_RangeSearch(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col=VEC_COL,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=TOPK,
            ef_search=EF_SEARCH,
            ti_threshold=threshold,
            output_file=output_file,
            metric=args.metric,
        )
    elif args.method == "loophnsw_range":
        loop_HNSWRange(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col=VEC_COL,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=-1,
            ef_search=EF_SEARCH,
            ti_threshold=threshold,
            output_file=output_file,
            metric=args.metric,
        )
    elif args.method == "lateralhnsw_range":
        laterCrossJoin_HNSWRange(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col=VEC_COL,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=-1,
            ef_search=EF_SEARCH,
            ti_threshold=threshold,
            output_file=output_file,
            metric=args.metric,
        )
    elif args.method == "directhnsw_range":
        direct_HNSWRange(
            conn=conn,
            table_left=TABLE_LEFT,
            table_right=TABLE_RIGHT,
            vec_col_left=VEC_COL_LEFT,
            vec_col_right=VEC_COL_RIGHT,
            id_col_left=ID_COL_LEFT,
            id_col_right=ID_COL_RIGHT,
            source_batch=SOURCE_BATCH,
            topk=-1,
            ef_search=EF_SEARCH,
            ti_threshold=threshold,
            output_file=output_file,
            metric=args.metric,
        )

    # materialize ti
    conn.autocommit = True
    materialize_ti(
        conn=conn,
        input_file=output_file,
        output_table=f"ti_{TABLE_LEFT}_{TABLE_RIGHT}_{threshold}",
        table_left=TABLE_LEFT,
        table_right=TABLE_RIGHT,
        id_col_left=ID_COL_LEFT,
        id_col_right=ID_COL_RIGHT,
        cols_left=wl["cols_left"],
        cols_right=wl["cols_right"],
    )


if __name__ == "__main__":
    main()
