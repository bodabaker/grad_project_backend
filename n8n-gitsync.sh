#!/usr/bin/env bash
# Plug-and-play n8n workflow backup/restore via the n8n CLI (no API key).
# Works with a running Docker container named "n8n".
# Requires: bash, docker (and/or docker compose), git

set -euo pipefail

# --- Configuration (override via env if needed) ---
: "${N8N_CONTAINER:=n8n}"                 # container name
: "${WORKFLOWS_DIR:=n8n/workflows}"       # path in your repo to store JSONs

# --- Helpers ---
die() { echo "❌ $*" >&2; exit 1; }

has_compose() {
  command -v docker-compose >/dev/null 2>&1 && return 0
  docker compose version >/dev/null 2>&1
}

is_running() {
  docker ps --format '{{.Names}}' | grep -qx "${N8N_CONTAINER}"
}

exec_in() {
  # Prefer `docker compose exec`, fall back to `docker exec`
  if has_compose; then
    docker compose exec -T "${N8N_CONTAINER}" bash -lc "$*"
  else
    docker exec -i "${N8N_CONTAINER}" bash -lc "$*"
  fi
}

copy_from() {
  docker cp "${N8N_CONTAINER}:$1" "$2"
}

copy_to() {
  docker cp "$1" "${N8N_CONTAINER}:$2"
}

ensure_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Run from your git repo root."
  mkdir -p "${WORKFLOWS_DIR}"
}

ensure_running() {
  is_running || die "Container '${N8N_CONTAINER}' not running. Start your stack first (e.g., 'docker compose up -d')."
}

backup_workflows() {
  ensure_running
  ensure_repo

  echo "📝 Exporting workflows from container '${N8N_CONTAINER}' ..."
  # Export inside container to /tmp/wf, then docker cp to host
  exec_in "rm -rf /tmp/wf && mkdir -p /tmp/wf && npx --yes n8n export:workflow --all --separate --pretty --output /tmp/wf"
  rm -rf ./.tmp_wf && mkdir -p ./.tmp_wf
  copy_from "/tmp/wf/." "./.tmp_wf"

  # Move into repo folder
  mkdir -p "${WORKFLOWS_DIR}"
  rsync -a --delete ./.tmp_wf/ "${WORKFLOWS_DIR}/"
  rm -rf ./.tmp_wf

  # Commit if changed
  git add -A "${WORKFLOWS_DIR}"
  if ! git diff --cached --quiet; then
    git commit -m "n8n CLI backup workflows: $(date -u +'%F %T UTC')"
    echo "✅ Workflows exported and committed."
  else
    echo "ℹ️ No changes to commit."
  fi
}

restore_workflows() {
  ensure_running
  ensure_repo

  # Ensure we have files to restore
  shopt -s nullglob
  set +e
  cnt=$(ls -1 "${WORKFLOWS_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')
  set -e
  [ "$cnt" -gt 0 ] || die "No JSON files in '${WORKFLOWS_DIR}'."

  echo "♻️ Importing workflows into container '${N8N_CONTAINER}' ..."
  # Copy host folder into container and import
  copy_to "${WORKFLOWS_DIR}" "/tmp/wf_in"
  # Note: Depending on n8n version, you may have --overwrite. If supported, add it to avoid duplicates.
  exec_in "npx --yes n8n import:workflow --separate --input /tmp/wf_in"
  echo "✅ Workflows imported."
}

git_push() {
  git push || die "git push failed. Set remote and credentials, then retry."
  echo "🚀 Pushed to remote."
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  backup      Export all n8n workflows (via CLI) to '${WORKFLOWS_DIR}' and commit
  restore     Import workflows from '${WORKFLOWS_DIR}' into n8n (via CLI)
  push        git push

Env overrides:
  N8N_CONTAINER=${N8N_CONTAINER}
  WORKFLOWS_DIR=${WORKFLOWS_DIR}

Notes:
- No n8n API key required. The script uses the n8n CLI inside the running container.
- This backs up WORKFLOWS only (safe for public repos).
- For credentials, use DB backups (SQLite) or a private repo if exporting decrypted creds.
EOF
}

cmd="${1:-help}"
case "$cmd" in
  backup)  backup_workflows;;
  restore) restore_workflows;;
  push)    git_push;;
  help|*)  usage;;
esac

