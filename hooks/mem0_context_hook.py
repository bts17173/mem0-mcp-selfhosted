#!/usr/bin/env python3
"""mem0 SessionStart hook — injects relevant memories as context.

Called by Claude Code at session start. Searches mem0 for memories
relevant to the current project/context and injects them as
additional context for the session.
"""

import json
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("mem0-context")


def hook_response(context: str = "") -> str:
    if context:
        return json.dumps({
            "continue": True,
            "suppressOutput": True,
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            },
        })
    return json.dumps({"continue": True, "suppressOutput": True})


def format_memories(results: list) -> str:
    """Format mem0 search results as injected context."""
    if not results:
        return ""

    lines = ["# mem0 Cross-Session Memory", ""]
    for i, mem in enumerate(results, 1):
        text = mem.get("memory", "")
        if text:
            lines.append(f"{i}. {text}")

    lines.append("")
    lines.append(f"({len(results)} memories loaded from mem0 persistent store)")
    return "\n".join(lines)


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

    cwd = data.get("cwd", "")

    # Build search query from project context
    project_name = os.path.basename(cwd) if cwd else ""
    query = f"project context and principles for {project_name}" if project_name else "personal principles and decisions"

    try:
        from mem0_init import get_memory, get_user_id

        memory = get_memory()
        user_id = get_user_id()

        results = memory.search(query=query, user_id=user_id, limit=15)

        if isinstance(results, dict) and "results" in results:
            memories = results["results"]
        elif isinstance(results, list):
            memories = results
        else:
            memories = []

        if memories:
            context = format_memories(memories)
            print(hook_response(context))
        else:
            print(hook_response())

    except Exception as e:
        logger.error("Failed to search mem0: %s", e)
        print(hook_response())


if __name__ == "__main__":
    main()
