# dase_batch_hnsw extension: batched HNSW range probes for SemJI construction.
# Compiles the traversal code of the DASE pgvector fork in ../pgvector/src.
#   make && sudo make install          (PG_CONFIG=/path/to/pg_config to pick a server)
EXTENSION = dase_batch_hnsw
MODULE_big = dase_batch_hnsw
OBJS = dase_batch_hnsw.o
DATA = dase_batch_hnsw--1.0.sql

PG_CONFIG ?= pg_config

# Same optimization flags as ../pgvector/Makefile, so the traversal code here
# matches the one compiled into vector.so
PG_CPPFLAGS = -I$(CURDIR)/../pgvector/src -I$(CURDIR)/compat
PG_CFLAGS += -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
