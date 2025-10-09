#!/usr/bin/env bash
# scripts/n8n-prune.sh
# Prune n8n executions + VACUUM SQLite inside the running container, then stage the DB.

set -euo pipefail

N8N_CONTAINER="${N8N_CONTAINER:-n8n}"
DB_PATH="${DB_PATH:-n8n_data/database.sqlite}"

# Skip quietly if container isn't running (don’t block commits)
if ! docker ps --format '{{.Names}}' | grep -qx "$N8N_CONTAINER"; then
  echo "[n8n-prune] Container '$N8N_CONTAINER' not running; skipping."
  exit 0
fi

echo "[n8n-prune] Pruning executions…"
docker exec -i "$N8N_CONTAINER" n8n prune executions || true

echo "[n8n-prune] VACUUM database…"
docker exec -i "$N8N_CONTAINER" sh -lc \
  'sqlite3 /home/node/.n8n/database.sqlite "PRAGMA wal_checkpoint(FULL); VACUUM;"' \
  || echo "[n8n-prune] sqlite3 not present; skipped VACUUM."

# Stage DB if it exists on host
if [ -f "$DB_PATH" ]; then
  echo "[n8n-prune] Staging $DB_PATH"
  git add "$DB_PATH"
fi
