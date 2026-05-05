-- Upgrade script from 0.8.4 to 0.8.5

DROP FUNCTION IF EXISTS hnsw_set_tid_map_table(text);

CREATE FUNCTION hnsw_set_tid_map_table(text) RETURNS void
	AS 'MODULE_PATHNAME', 'hnsw_set_tid_map_table_sql' LANGUAGE C VOLATILE;

