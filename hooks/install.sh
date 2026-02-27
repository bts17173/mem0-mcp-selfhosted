#!/bin/bash
# mem0 hooks installer — registers mem0 hooks in a Claude Code project
#
# Usage:
#   bash /path/to/mem0-mcp-selfhosted/hooks/install.sh [project_dir]
#
# If project_dir is omitted, uses current directory.
# Creates/updates .claude/settings.json with mem0 hook configuration.
#
# Installs:
#   SessionStart → mem0 context injection (search + inject relevant memories)
#   Stop         → mem0 session summary (LLM extracts key decisions/lessons)
#
# NOT installed (by design):
#   PostToolUse  → Intentionally omitted. Continuous recording creates noise
#                  without a self-correction mechanism. mem0's design philosophy
#                  is curated quality over automated quantity.

set -euo pipefail

MEM0_HOME="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="${1:-$(pwd)}"
SETTINGS_DIR="$PROJECT_DIR/.claude"
SETTINGS_FILE="$SETTINGS_DIR/settings.json"

echo "mem0 hooks installer"
echo "  MEM0_HOME:    $MEM0_HOME"
echo "  PROJECT_DIR:  $PROJECT_DIR"
echo ""

# Verify mem0 installation
if [ ! -f "$MEM0_HOME/.venv/bin/python3" ]; then
    echo "ERROR: mem0 venv not found at $MEM0_HOME/.venv/"
    echo "Run: cd $MEM0_HOME && python3 -m venv .venv && pip install -e ."
    exit 1
fi

# Verify hooks exist
for script in mem0-context-hook.sh mem0-stop-hook.sh; do
    if [ ! -x "$MEM0_HOME/hooks/$script" ]; then
        echo "ERROR: $script not found or not executable"
        exit 1
    fi
done

# Create .claude dir if needed
mkdir -p "$SETTINGS_DIR"

# Generate hook config JSON
CONTEXT_HOOK="$MEM0_HOME/hooks/mem0-context-hook.sh"
STOP_HOOK="$MEM0_HOME/hooks/mem0-stop-hook.sh"

if [ -f "$SETTINGS_FILE" ]; then
    # Merge into existing settings
    python3 -c "
import json

with open('$SETTINGS_FILE', 'r') as f:
    settings = json.load(f)

hooks = settings.setdefault('hooks', {})

# SessionStart hook
session_hooks = hooks.setdefault('SessionStart', [])
mem0_context_cmd = '$CONTEXT_HOOK'
already = any(
    any(h.get('command', '') == mem0_context_cmd for h in entry.get('hooks', []))
    for entry in session_hooks
)
if not already:
    session_hooks.append({
        'matcher': 'startup|clear|compact',
        'hooks': [{'type': 'command', 'command': mem0_context_cmd, 'timeout': 30000}]
    })

# Stop hook
stop_hooks = hooks.setdefault('Stop', [])
mem0_stop_cmd = '$STOP_HOOK'
already = any(
    any(h.get('command', '') == mem0_stop_cmd for h in entry.get('hooks', []))
    for entry in stop_hooks
)
if not already:
    stop_hooks.append({
        'hooks': [{'type': 'command', 'command': mem0_stop_cmd, 'timeout': 60000}]
    })

# Disable claude-mem if present
plugins = settings.setdefault('enabledPlugins', {})
plugins['claude-mem@thedotmack'] = False

with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2, ensure_ascii=False)
    f.write('\n')

print('Updated: $SETTINGS_FILE')
"
else
    # Create new settings file
    cat > "$SETTINGS_FILE" << SETTINGS
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "$CONTEXT_HOOK",
            "timeout": 30000
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$STOP_HOOK",
            "timeout": 60000
          }
        ]
      }
    ]
  },
  "enabledPlugins": {
    "claude-mem@thedotmack": false
  }
}
SETTINGS
    echo "Created: $SETTINGS_FILE"
fi

echo ""
echo "Done! mem0 hooks installed:"
echo "  SessionStart → mem0 context injection (search relevant memories)"
echo "  Stop         → mem0 session summary (extract key decisions/lessons)"
echo "  claude-mem   → disabled"
echo ""
echo "Memory recording strategy: Stop-only + manual add_memory"
echo "  (No PostToolUse — quality over quantity, no noise accumulation)"
echo ""
echo "To uninstall: remove the mem0 hook entries from $SETTINGS_FILE"
