#!/usr/bin/env python3
"""mem0 Stop hook — saves session summary to mem0.

Called by Claude Code at session end. Reads hook input from stdin,
extracts the last user + assistant messages from the transcript,
and saves a concise session summary to mem0.

Unlike claude-mem's PostToolUse (every tool call = noise), this only
records once per session with high signal content.
"""

import json
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("mem0-stop")


def read_transcript_message(transcript_path: str, role: str, strip_system: bool = False) -> str:
    """Read the last message of a given role from the transcript JSONL."""
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    try:
        with open(transcript_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == role and entry.get("message", {}).get("content"):
                    content = entry["message"]["content"]
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = "\n".join(
                            block["text"] for block in content if block.get("type") == "text"
                        )
                    else:
                        continue
                    if strip_system:
                        import re
                        text = re.sub(r"<system-reminder>[\s\S]*?</system-reminder>", "", text)
                        text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    return text
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception as e:
        logger.warning("Failed to read transcript: %s", e)
    return ""


def build_summary_prompt(last_user: str, last_assistant: str, cwd: str) -> str:
    """Build a prompt that extracts key session learnings."""
    parts = []
    if cwd:
        parts.append(f"Project: {cwd}")
    if last_user:
        parts.append(f"Last user message:\n{last_user[:2000]}")
    if last_assistant:
        parts.append(f"Last assistant message:\n{last_assistant[:3000]}")

    context = "\n\n".join(parts)

    return f"""Based on this session ending context, extract ONLY the key decisions,
lessons learned, or important facts worth remembering long-term.

{context}

Output a concise summary (2-5 sentences) focusing on:
- Decisions made and why
- Problems solved and the solution approach
- Important patterns or principles discovered
- Configuration changes with rationale

Skip trivial actions (file reads, routine commits). Only capture what would
help a future session understand context or avoid repeating mistakes."""


def hook_response(success: bool = True) -> str:
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

    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    cwd = data.get("cwd", "")

    last_user = read_transcript_message(transcript_path, "user")
    last_assistant = read_transcript_message(transcript_path, "assistant", strip_system=True)

    if not last_user and not last_assistant:
        logger.info("No transcript content, skipping mem0 save")
        print(hook_response())
        return

    # Truncate very short sessions (likely just a greeting or single command)
    if len(last_user) < 20 and len(last_assistant) < 50:
        logger.info("Session too short, skipping mem0 save")
        print(hook_response())
        return

    try:
        from mem0_init import get_memory, get_user_id

        memory = get_memory()
        user_id = get_user_id()

        summary_text = build_summary_prompt(last_user, last_assistant, cwd)

        # Use mem0's infer=True to auto-extract atomic facts
        memory.add(
            [{"role": "user", "content": summary_text}],
            user_id=user_id,
            metadata={"source": "session-stop-hook", "session_id": session_id},
        )
        logger.info("Session summary saved to mem0")
    except Exception as e:
        logger.error("Failed to save to mem0: %s", e)

    print(hook_response())


if __name__ == "__main__":
    main()
