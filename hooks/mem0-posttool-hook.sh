#!/bin/bash
# mem0 PostToolUse hook — selective recording of write operations
# Portable: uses MEM0_HOME env var or auto-detects from script location

MEM0_HOME="${MEM0_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_PYTHON="$MEM0_HOME/.venv/bin/python3"
HOOKS_DIR="$MEM0_HOME/hooks"

[ -f "$MEM0_HOME/.env" ] && { set -a; source "$MEM0_HOME/.env"; set +a; }

exec "$VENV_PYTHON" "$HOOKS_DIR/mem0_posttool_hook.py"
