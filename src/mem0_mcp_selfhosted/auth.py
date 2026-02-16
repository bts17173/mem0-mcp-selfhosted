"""Hybrid Anthropic token resolution with fallback chain.

Fallback order:
1. MEM0_ANTHROPIC_TOKEN env var
2. ~/.claude/.credentials.json (claudeAiOauth.accessToken)
3. ANTHROPIC_API_KEY env var
4. None (disabled with warning)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def is_oat_token(token: str) -> bool:
    """Detect whether the token is an OAT token (vs standard API key)."""
    return "sk-ant-oat" in token


def _read_credentials_file() -> str | None:
    """Read accessToken from ~/.claude/.credentials.json.

    Returns the token string or None. Handles:
    - Missing file (silent — file absence is expected)
    - Malformed JSON (warns)
    - Missing accessToken key (warns)
    """
    if not _CREDENTIALS_PATH.exists():
        return None

    try:
        data = json.loads(_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse credentials file %s: %s", _CREDENTIALS_PATH, exc)
        return None

    try:
        token = data["claudeAiOauth"]["accessToken"]
    except (KeyError, TypeError):
        logger.warning(
            "Credentials file %s missing claudeAiOauth.accessToken", _CREDENTIALS_PATH
        )
        return None

    if not token or not isinstance(token, str):
        logger.warning("Credentials file accessToken is empty or invalid")
        return None

    return token


def resolve_token() -> str | None:
    """Resolve an Anthropic auth token using the prioritized fallback chain.

    Returns the resolved token or None if no auth is available.
    """
    # Priority 1: Explicit env var
    token = os.environ.get("MEM0_ANTHROPIC_TOKEN")
    if token:
        token_type = "OAT" if is_oat_token(token) else "API key"
        logger.debug("Auth resolved from MEM0_ANTHROPIC_TOKEN (type: %s)", token_type)
        return token

    # Priority 2: Claude Code credentials file
    token = _read_credentials_file()
    if token:
        token_type = "OAT" if is_oat_token(token) else "API key"
        logger.debug(
            "Auth resolved from %s (type: %s)", _CREDENTIALS_PATH, token_type
        )
        return token

    # Priority 3: Standard API key
    token = os.environ.get("ANTHROPIC_API_KEY")
    if token:
        token_type = "OAT" if is_oat_token(token) else "API key"
        logger.debug("Auth resolved from ANTHROPIC_API_KEY (type: %s)", token_type)
        return token

    # No auth available
    logger.warning(
        "No Anthropic token found. Checked: MEM0_ANTHROPIC_TOKEN env var, "
        "%s, ANTHROPIC_API_KEY env var. "
        "Anthropic LLM features will be disabled.",
        _CREDENTIALS_PATH,
    )
    return None
