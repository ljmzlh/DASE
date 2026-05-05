-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.8.2'" to load this file. \quit

-- Fix vector_filter_sql to match vector_within_radius_sql logic and remove PL/pgSQL raise guard
CREATE OR REPLACE FUNCTION vector_filter_sql(vec vector, query vector) RETURNS boolean
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $body$
    SELECT ($1 <-> $2) <= current_setting('hnsw.radius')::float8
$body$;
