"""Shared mem0 Memory initializer for hook scripts.

Reads config from env vars (same as MCP server).
Hooks disable graph by default (Neo4j failure = non-fatal).
Designed to be portable — can be installed in any project.
"""

import os
import sys
import logging

# Add parent src to path for config reuse
_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

logger = logging.getLogger("mem0-hooks")

_memory = None


def get_memory(enable_graph: bool = False):
    """Lazy-init mem0 Memory. Graph disabled by default for hooks (fast + resilient)."""
    global _memory
    if _memory is not None:
        return _memory

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    # Force graph off in hooks unless explicitly requested
    if not enable_graph:
        os.environ["MEM0_ENABLE_GRAPH"] = "false"

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.helpers import patch_graph_sanitizer

    config_dict, providers_info, _split_config = build_config()

    if providers_info:
        from mem0.utils.factory import LlmFactory
        for pi in providers_info:
            if pi["name"] == "openai":
                from mem0.configs.llms.openai import OpenAIConfig
                LlmFactory.register_provider(pi["name"], pi["class_path"], OpenAIConfig)

    patch_graph_sanitizer()

    from mem0 import Memory
    _memory = Memory.from_config(config_dict)

    return _memory


def get_user_id():
    return os.environ.get("MEM0_USER_ID", "wuwei")
