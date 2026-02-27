#!/usr/bin/env python3
"""mem0 PostToolUse hook — selective recording of significant operations.

Only records write operations (Write, Edit, Bash commits) to mem0.
Skips reads, searches, and other non-modifying operations.
This is the key difference from claude-mem: quality > quantity.
"""

import json
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("mem0-posttool")

# Tools worth recording (write operations only)
RECORD_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

# Bash commands worth recording
RECORD_BASH_PATTERNS = [
    "git commit",
    "git merge",
    "git rebase",
    "npm install",
    "pip install",
    "bun install",
]

# Skip these tools entirely
SKIP_TOOLS = {
    "Read", "Glob", "Grep", "LSP", "WebFetch", "WebSearch",
    "Task", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "AskUserQuestion", "EnterPlanMode", "ExitPlanMode",
    "Skill", "ListMcpResourcesTool", "ReadMcpResourceTool",
}


def should_record(tool_name: str, tool_input: str) -> bool:
    """Decide if this tool use is worth recording."""
    if tool_name in SKIP_TOOLS:
        return False

    if tool_name in RECORD_TOOLS:
        return True

    if tool_name == "Bash":
        try:
            input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            command = input_data.get("command", "")
            return any(pat in command for pat in RECORD_BASH_PATTERNS)
        except (json.JSONDecodeError, AttributeError):
            return False

    # MCP tools — skip browser/playwright, record mem0 writes
    if tool_name.startswith("mcp__Playwright") or tool_name.startswith("mcp__claude-in-chrome"):
        return False
    if tool_name.startswith("mcp__mem0__add_memory") or tool_name.startswith("mcp__mem0__update_memory"):
        return False  # Don't record mem0 writes to avoid recursion

    return False


def build_observation_text(tool_name: str, tool_input: str, cwd: str) -> str:
    """Build a concise description of the operation."""
    try:
        input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
    except (json.JSONDecodeError, TypeError):
        input_data = {}

    if tool_name in ("Write", "Edit", "MultiEdit"):
        file_path = input_data.get("file_path", "unknown")
        rel_path = os.path.relpath(file_path, cwd) if cwd and file_path != "unknown" else file_path
        if tool_name == "Write":
            return f"Created/overwrote file: {rel_path}"
        elif tool_name == "Edit":
            old = (input_data.get("old_string", ""))[:80]
            new = (input_data.get("new_string", ""))[:80]
            return f"Edited {rel_path}: '{old}' → '{new}'"
        else:
            return f"Multi-edited file: {rel_path}"

    if tool_name == "Bash":
        command = input_data.get("command", "")[:200]
        return f"Ran command: {command}"

    return f"{tool_name}: {str(input_data)[:200]}"


def hook_response() -> str:
    return json.dumps({"continue": True, "suppressOutput": True})


def main():
    raw = sys.stdin.read()
    if not raw:
        print(hook_response())
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(hook_response())
        return

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", "")
    session_id = data.get("session_id", "")
    cwd = data.get("cwd", "")

    if not should_record(tool_name, tool_input):
        print(hook_response())
        return

    observation = build_observation_text(tool_name, tool_input, cwd)
    logger.info("Recording: %s", observation[:100])

    try:
        from mem0_init import get_memory, get_user_id

        memory = get_memory()
        user_id = get_user_id()

        memory.add(
            [{"role": "user", "content": observation}],
            user_id=user_id,
            metadata={
                "source": "posttool-hook",
                "tool": tool_name,
                "session_id": session_id,
            },
            infer=False,  # Store raw — this is a log entry, not a knowledge extraction
        )
    except Exception as e:
        logger.error("Failed to record to mem0: %s", e)

    print(hook_response())


if __name__ == "__main__":
    main()
