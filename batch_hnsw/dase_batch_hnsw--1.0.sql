-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION dase_batch_hnsw" to load this file. \quit

-- Objects land in schema bh (see dase_batch_hnsw.control).

-- Probe right_index once per row of left_sql and return every pair within
-- radius. left_sql returns (id int8, vec vector, grp int4) in processing order;
-- mode is probe, share or warm (see dase_batch_hnsw.c).
CREATE FUNCTION range_join_all(
    left_sql text, right_index regclass, radius float8, ef int4,
    mode text, n_seeds int4)
RETURNS TABLE (left_id int8, right_tid tid, dist float8)
AS 'MODULE_PATHNAME', 'bh_range_join_all' LANGUAGE C STRICT;

-- JSON with the counts and timings of this backend's last range_join_all call
CREATE FUNCTION last_stats() RETURNS text
AS 'MODULE_PATHNAME', 'bh_last_stats' LANGUAGE C STRICT;
