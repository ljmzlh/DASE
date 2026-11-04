# syntax=docker/dockerfile:1
#
# DASE in one image: PostgreSQL 18 with the DASE pgvector fork and the
# dase_batch_hnsw extension, plus this repository and its Python dependencies.
# The container runs the server; DASE commands run through `docker exec`.
# See README §1.3.
#
#   docker build -t dase .

ARG PG_IMAGE=postgres:18.6-trixie

FROM ${PG_IMAGE} AS build
# Empty = portable binaries (no -march=native); the vector distance functions
# still pick FMA at run time.
ARG OPTFLAGS=""
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential postgresql-server-dev-${PG_MAJOR} \
 && rm -rf /var/lib/apt/lists/*
COPY pgvector /src/pgvector
COPY batch_hnsw /src/batch_hnsw
RUN make -C /src/pgvector clean \
 && make -C /src/pgvector OPTFLAGS="${OPTFLAGS}" \
 && make -C /src/pgvector install DESTDIR=/out \
 && make -C /src/batch_hnsw OPTFLAGS="${OPTFLAGS}" \
 && make -C /src/batch_hnsw install DESTDIR=/out

FROM ${PG_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /

COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

COPY docker/dase.conf /etc/postgresql/dase.conf
# --chmod: a downloaded archive of the repository may not keep the execute bits.
COPY --chmod=755 docker/initdb-dase.sh /docker-entrypoint-initdb.d/10-dase.sh
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/dase-entrypoint.sh

# The code connects to postgresql://localhost/<db> without a user name;
# PGUSER makes that the superuser the official image creates.
ENV PATH=/opt/venv/bin:$PATH \
    PGUSER=postgres \
    PYTHONUNBUFFERED=1

COPY . /dase
WORKDIR /dase

ENTRYPOINT ["dase-entrypoint.sh"]
CMD ["postgres"]
