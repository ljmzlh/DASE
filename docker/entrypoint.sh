#!/bin/sh
# The official entrypoint refuses to create a cluster without a superuser
# password. Connections from inside the container are trusted and never use
# it, so default to a random one; set POSTGRES_PASSWORD to connect from outside.
set -e
if [ -z "${POSTGRES_PASSWORD:-}" ]; then
	POSTGRES_PASSWORD=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
	export POSTGRES_PASSWORD
fi
exec docker-entrypoint.sh "$@"
