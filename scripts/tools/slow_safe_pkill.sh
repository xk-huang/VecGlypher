#!/bin/bash
# e.g.,
DOCSTRING='
bash scripts/tools/slow_safe_pkill.sh "vllm"
bash scripts/tools/slow_safe_pkill.sh "VLLM"
bash scripts/tools/slow_safe_pkill.sh "multiprocessing.spawn"
'
set -e

slow_safe_pkill() {
  sleep 5
  local pattern="$1"
  local user="${2:-"$(whoami)"}"

  # 1. Try TERM
  pkill -u "$user" -f -TERM "$pattern" 2>/dev/null || true
  sleep 5

  # 2. If still alive, try INT (like Ctrl+C)
  if pgrep -u "$user" -f "$pattern" >/dev/null; then
    pkill -u "$user" -f -INT "$pattern" 2>/dev/null || true
    sleep 5
  fi

  # 3. If still alive, force KILL
  if pgrep -u "$user" -f "$pattern" >/dev/null; then
    pkill -u "$user" -f -9 "$pattern" 2>/dev/null || true
    sleep 5
  fi
}

slow_safe_pkill "$@"
