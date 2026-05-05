#!/usr/bin/env bash
set -euo pipefail

# Override these via environment variables.
PG_CONFIG="${PG_CONFIG:-/usr/lib/postgresql/14/bin/pg_config}"
DB="${DB:-molecule}"
PSQL_USER="${PSQL_USER:-postgres}"

echo "[1/3] make clean"
make clean

echo "[2/3] make (PG_CONFIG=$PG_CONFIG)"
make PG_CONFIG="$PG_CONFIG"

echo "[3/3] sudo make install (PG_CONFIG=$PG_CONFIG)"
sudo make install PG_CONFIG="$PG_CONFIG"

echo "[OK] pgvector reinstalled. Open a NEW psql session to load new .so."
echo "Hint: psql -U ${PSQL_USER} -d ${DB}"
echo
echo "[Optional] Setting client_min_messages=NOTICE in this one-off session..."
psql -U "${PSQL_USER}" -d "${DB}" -c "SET client_min_messages = NOTICE;" || true

