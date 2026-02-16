"""Environment-driven configuration for mem0-mcp-selfhosted.

Reads all config from env vars with sensible defaults, constructs a
mem0ai MemoryConfig dict, and returns provider registration info.
"""

from __future__ import annotations

import os
from typing import Any

from mem0_mcp_selfhosted.auth import resolve_token


def _bool_env(key: str, default: str = "false") -> bool:
    return os.environ.get(key, default).lower() in ("true", "1", "yes")


def build_config() -> tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]:
    """Build mem0ai MemoryConfig dict and provider registration info.

    Returns:
        (config_dict, provider_info, split_config) where:
        - provider_info: primary provider dict with name, class_path, config_class_path
        - split_config: if gemini_split was requested, config for the SplitModelGraphLLM
    """
    token = resolve_token()

    # --- LLM ---
    llm_model = os.environ.get("MEM0_LLM_MODEL", "claude-opus-4-6")
    llm_max_tokens = int(os.environ.get("MEM0_LLM_MAX_TOKENS", "16384"))

    llm_config: dict[str, Any] = {
        "model": llm_model,
        "max_tokens": llm_max_tokens,
    }
    if token:
        llm_config["api_key"] = token

    # --- Embedder ---
    embed_provider = os.environ.get("MEM0_EMBED_PROVIDER", "ollama")
    embed_model = os.environ.get("MEM0_EMBED_MODEL", "bge-m3")
    embed_url = os.environ.get("MEM0_EMBED_URL", "http://localhost:11434")
    embed_dims = int(os.environ.get("MEM0_EMBED_DIMS", "1024"))

    embedder_config: dict[str, Any] = {
        "model": embed_model,
    }
    if embed_provider == "ollama":
        embedder_config["ollama_base_url"] = embed_url

    # --- Vector Store ---
    qdrant_url = os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("MEM0_COLLECTION", "mem0_mcp_selfhosted")
    qdrant_api_key = os.environ.get("MEM0_QDRANT_API_KEY")
    qdrant_on_disk = _bool_env("MEM0_QDRANT_ON_DISK")

    vector_config: dict[str, Any] = {
        "collection_name": collection,
        "url": qdrant_url,
        "embedding_model_dims": embed_dims,
    }
    if qdrant_api_key:
        vector_config["api_key"] = qdrant_api_key
    if qdrant_on_disk:
        vector_config["on_disk"] = True

    # --- History ---
    history_db_path = os.environ.get("MEM0_HISTORY_DB_PATH")

    # --- Build config dict ---
    config_dict: dict[str, Any] = {
        "llm": {
            "provider": "anthropic",
            "config": llm_config,
        },
        "embedder": {
            "provider": embed_provider,  # Explicit — never rely on mem0ai's openai default
            "config": embedder_config,
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_config,
        },
        "version": "v1.1",
    }

    if history_db_path:
        config_dict["history_db_path"] = history_db_path

    # --- Graph Store (conditional) ---
    enable_graph = _bool_env("MEM0_ENABLE_GRAPH")
    if enable_graph:
        neo4j_url = os.environ.get("MEM0_NEO4J_URL", "bolt://127.0.0.1:7687")
        neo4j_user = os.environ.get("MEM0_NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("MEM0_NEO4J_PASSWORD", "mem0graph")
        neo4j_database = os.environ.get("MEM0_NEO4J_DATABASE")
        neo4j_base_label = os.environ.get("MEM0_NEO4J_BASE_LABEL")
        graph_threshold = float(os.environ.get("MEM0_GRAPH_THRESHOLD", "0.7"))

        graph_neo4j_config: dict[str, Any] = {
            "url": neo4j_url,
            "username": neo4j_user,
            "password": neo4j_password,
        }
        if neo4j_database:
            graph_neo4j_config["database"] = neo4j_database
        if neo4j_base_label:
            graph_neo4j_config["base_label"] = neo4j_base_label

        # Graph LLM — MUST be explicit (mem0ai defaults to "openai" if omitted)
        graph_llm_provider_raw = os.environ.get("MEM0_GRAPH_LLM_PROVIDER", "anthropic")
        graph_llm_provider = graph_llm_provider_raw
        graph_llm_model = os.environ.get("MEM0_GRAPH_LLM_MODEL", llm_model)

        graph_llm_config: dict[str, Any] = {
            "model": graph_llm_model,
        }

        if graph_llm_provider == "ollama":
            ollama_url = os.environ.get("MEM0_EMBED_URL", "http://localhost:11434")
            graph_llm_config["ollama_base_url"] = ollama_url
        elif graph_llm_provider in ("anthropic", "anthropic_oat"):
            if token:
                graph_llm_config["api_key"] = token
            graph_llm_config["max_tokens"] = llm_max_tokens
        elif graph_llm_provider == "gemini":
            # Use mem0ai's built-in GeminiLLM provider
            # Default to flash-lite (not the main Claude model) when no explicit model set
            graph_llm_config["model"] = os.environ.get(
                "MEM0_GRAPH_LLM_MODEL", "gemini-2.5-flash-lite"
            )
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if google_api_key:
                graph_llm_config["api_key"] = google_api_key
        elif graph_llm_provider == "gemini_split":
            # Split-model router: Gemini for extraction, separate LLM for contradiction.
            # Use "gemini" as config provider (passes pydantic validation), then
            # server.py swaps the graph LLM to the SplitModelGraphLLM after creation.
            graph_llm_config["model"] = os.environ.get(
                "MEM0_GRAPH_LLM_MODEL", "gemini-2.5-flash-lite"
            )
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if google_api_key:
                graph_llm_config["api_key"] = google_api_key
            # Override provider to "gemini" for pydantic validation
            graph_llm_provider = "gemini"

        config_dict["graph_store"] = {
            "provider": "neo4j",
            "config": graph_neo4j_config,
            "threshold": graph_threshold,
            "llm": {
                "provider": graph_llm_provider,
                "config": graph_llm_config,
            },
        }

    # --- Provider registration info ---
    provider_info = {
        "name": "anthropic",
        "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
        "config_class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATConfig",
    }

    # Split-model config: if gemini_split was requested, provide the config
    # for server.py to swap the graph LLM after Memory creation.
    split_config: dict[str, Any] | None = None
    if enable_graph and graph_llm_provider_raw == "gemini_split":
        extraction_model = os.environ.get("MEM0_GRAPH_LLM_MODEL", "gemini-2.5-flash-lite")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        contradiction_provider = os.environ.get(
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER", "anthropic"
        )
        contradiction_model = os.environ.get(
            "MEM0_GRAPH_CONTRADICTION_LLM_MODEL", llm_model
        )
        split_config = {
            "extraction_provider": "gemini",
            "extraction_model": extraction_model,
            "contradiction_provider": contradiction_provider,
            "contradiction_model": contradiction_model,
            "contradiction_max_tokens": llm_max_tokens,
        }
        if google_api_key:
            split_config["extraction_api_key"] = google_api_key
        if contradiction_provider in ("anthropic", "anthropic_oat") and token:
            split_config["contradiction_api_key"] = token
        elif contradiction_provider == "ollama":
            split_config["contradiction_ollama_base_url"] = os.environ.get(
                "MEM0_EMBED_URL", "http://localhost:11434"
            )

    return config_dict, provider_info, split_config
