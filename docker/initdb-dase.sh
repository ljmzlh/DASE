#!/bin/sh
# Runs once, when the official entrypoint creates the cluster. Settings changed
# later with ALTER SYSTEM take precedence over the included file.
echo "include_if_exists = '/etc/postgresql/dase.conf'" >> "$PGDATA/postgresql.conf"
