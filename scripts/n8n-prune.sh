#!/usr/bin/env bash
# scripts/n8n-prune.sh
# Prune n8n logs + old executions from SQLite, then VACUUM.
# Compatible with n8n builds that do not expose `n8n prune executions`.

set -euo pipefail

N8N_CONTAINER="${N8N_CONTAINER:-n8n}"
DB_PATH="${DB_PATH:-n8n_data/database.sqlite}"
KEEP_EXECUTIONS="${KEEP_EXECUTIONS:-300}"
TRUNCATE_EVENT_LOGS="${TRUNCATE_EVENT_LOGS:-1}"
RESTART_N8N="${RESTART_N8N:-1}"
STAGE_DB="${STAGE_DB:-0}"
PURGE_ALL_EXECUTIONS="${PURGE_ALL_EXECUTIONS:-0}"
REMOVE_BINARY_DATA="${REMOVE_BINARY_DATA:-0}"
UNTRACK_RUNTIME_FILES="${UNTRACK_RUNTIME_FILES:-0}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[n8n-prune] python3 is required." >&2
  exit 1
fi

if [ ! -f "$DB_PATH" ]; then
  echo "[n8n-prune] DB not found at '$DB_PATH'; skipping."
  exit 0
fi

if ! [[ "$KEEP_EXECUTIONS" =~ ^[0-9]+$ ]]; then
  echo "[n8n-prune] KEEP_EXECUTIONS must be a non-negative integer." >&2
  exit 1
fi

for flag_name in PURGE_ALL_EXECUTIONS REMOVE_BINARY_DATA UNTRACK_RUNTIME_FILES; do
  flag_value="${!flag_name}"
  if ! [[ "$flag_value" =~ ^[01]$ ]]; then
    echo "[n8n-prune] $flag_name must be 0 or 1." >&2
    exit 1
  fi
done

DB_DIR="$(dirname "$DB_PATH")"

container_running="0"
if docker ps --format '{{.Names}}' | grep -qx "$N8N_CONTAINER"; then
  container_running="1"
fi

if [ "$TRUNCATE_EVENT_LOGS" = "1" ]; then
  echo "[n8n-prune] Truncating n8n event logs..."
  for f in "$DB_DIR"/n8nEventLog*.log; do
    [ -e "$f" ] || continue
    : > "$f"
  done
fi

if [ "$container_running" = "1" ] && [ "$RESTART_N8N" = "1" ]; then
  echo "[n8n-prune] Stopping container '$N8N_CONTAINER' to release DB lock..."
  docker compose -f "$COMPOSE_FILE" stop "$N8N_CONTAINER"
fi

if [ "$PURGE_ALL_EXECUTIONS" = "1" ]; then
  echo "[n8n-prune] Purging all executions and execution payloads..."
else
  echo "[n8n-prune] Pruning execution rows (keep last $KEEP_EXECUTIONS)..."
fi
python3 - <<PY
import sqlite3

db_path = r"$DB_PATH"
keep = int(r"$KEEP_EXECUTIONS")
purge_all = r"$PURGE_ALL_EXECUTIONS" == "1"

con = sqlite3.connect(db_path)
cur = con.cursor()
cur.execute("PRAGMA foreign_keys=ON;")
cur.execute("PRAGMA journal_mode=DELETE;")

max_id = cur.execute("SELECT COALESCE(MAX(id), 0) FROM execution_entity").fetchone()[0]
if purge_all:
  d_data = cur.execute("DELETE FROM execution_data").rowcount
  d_ent = cur.execute("DELETE FROM execution_entity").rowcount
  cutoff = max_id
else:
  cutoff = max(0, max_id - keep)
  if cutoff > 0:
    d_data = cur.execute("DELETE FROM execution_data WHERE executionId <= ?", (cutoff,)).rowcount
    d_ent = cur.execute("DELETE FROM execution_entity WHERE id <= ?", (cutoff,)).rowcount
  else:
    d_data = 0
    d_ent = 0

con.commit()
cur.execute("PRAGMA wal_checkpoint(FULL);")
cur.execute("VACUUM;")
con.commit()

remaining_ent = cur.execute("SELECT COUNT(*) FROM execution_entity").fetchone()[0]
remaining_data = cur.execute("SELECT COUNT(*) FROM execution_data").fetchone()[0]
con.close()

print(f"[n8n-prune] max_id={max_id}, cutoff={cutoff}")
print(f"[n8n-prune] deleted execution_entity={d_ent}, execution_data={d_data}")
print(f"[n8n-prune] remaining execution_entity={remaining_ent}, execution_data={remaining_data}")
PY

rm -f "$DB_DIR"/database.sqlite-wal "$DB_DIR"/database.sqlite-shm

if [ "$REMOVE_BINARY_DATA" = "1" ] && [ -d "$DB_DIR/binaryData" ]; then
  echo "[n8n-prune] Removing binary data files from $DB_DIR/binaryData ..."
  find "$DB_DIR/binaryData" -mindepth 1 -exec rm -rf {} +
fi

if [ "$container_running" = "1" ] && [ "$RESTART_N8N" = "1" ]; then
  echo "[n8n-prune] Restarting container '$N8N_CONTAINER'..."
  docker compose -f "$COMPOSE_FILE" up -d "$N8N_CONTAINER"
fi

if [ "$STAGE_DB" = "1" ] && [ -f "$DB_PATH" ]; then
  echo "[n8n-prune] Staging $DB_PATH"
  git add "$DB_PATH" || echo "[n8n-prune] git add failed; skipped."
fi

if [ "$UNTRACK_RUNTIME_FILES" = "1" ] && command -v git >/dev/null 2>&1; then
  echo "[n8n-prune] Removing runtime n8n artifacts from git index (local files stay on disk)..."
  git rm -r --cached --ignore-unmatch "$DB_DIR/binaryData" >/dev/null 2>&1 || true
  git rm --cached --ignore-unmatch "$DB_DIR/database.sqlite-wal" "$DB_DIR/database.sqlite-shm" >/dev/null 2>&1 || true
  git rm -r --cached --ignore-unmatch "$DB_DIR"/n8nEventLog*.log >/dev/null 2>&1 || true
fi

echo "[n8n-prune] Done."
