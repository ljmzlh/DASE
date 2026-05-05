-- Upgrade script from 0.8.3 to 0.8.4

CREATE OR REPLACE FUNCTION vector_filter_bytea_sql(vec vector, filter_map bytea) RETURNS boolean
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

DROP OPERATOR IF EXISTS <-># (vector, bytea);
CREATE OPERATOR <-># (
    LEFTARG = vector,
    RIGHTARG = bytea,
    PROCEDURE = vector_filter_bytea_sql,
    COMMUTATOR = '<->#'
);

ALTER OPERATOR FAMILY vector_l2_ops USING hnsw
    ADD OPERATOR 52 <-># (vector, bytea);

