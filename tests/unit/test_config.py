"""Tests for config.py — build_config() with various env var combinations."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestBuildConfig:
    def _build_with_env(self, env: dict):
        """Build config with the given env vars, mocking resolve_token."""
        # Clear env vars that could leak from integration tests or CLI
        leak_keys = [k for k in os.environ if k.startswith("MEM0_")]
        if "GOOGLE_API_KEY" in os.environ:
            leak_keys.append("GOOGLE_API_KEY")
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in leak_keys:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-test-token"):
                from mem0_mcp_selfhosted.config import build_config
                config_dict, provider_info, extra_providers = build_config()
                return config_dict, provider_info, extra_providers

    def test_defaults(self):
        """All defaults applied when no env vars set."""
        config_dict, provider_info, *_ = self._build_with_env({})

        assert config_dict["llm"]["provider"] == "anthropic"
        assert config_dict["llm"]["config"]["model"] == "claude-opus-4-6"
        assert config_dict["embedder"]["provider"] == "ollama"
        assert config_dict["embedder"]["config"]["model"] == "bge-m3"
        assert config_dict["vector_store"]["provider"] == "qdrant"
        assert config_dict["vector_store"]["config"]["collection_name"] == "mem0_mcp_selfhosted"
        assert "graph_store" not in config_dict
        assert config_dict["version"] == "v1.1"

    def test_env_overrides(self):
        """Environment variables override defaults."""
        env = {
            "MEM0_LLM_MODEL": "claude-sonnet-4-5-20250929",
            "MEM0_EMBED_MODEL": "nomic-embed-text",
            "MEM0_COLLECTION": "custom_collection",
        }
        config_dict, *_ = self._build_with_env(env)

        assert config_dict["llm"]["config"]["model"] == "claude-sonnet-4-5-20250929"
        assert config_dict["embedder"]["config"]["model"] == "nomic-embed-text"
        assert config_dict["vector_store"]["config"]["collection_name"] == "custom_collection"

    def test_graph_enabled(self):
        """Graph store included when MEM0_ENABLE_GRAPH=true."""
        env = {"MEM0_ENABLE_GRAPH": "true"}
        config_dict, *_ = self._build_with_env(env)

        assert "graph_store" in config_dict
        assert config_dict["graph_store"]["provider"] == "neo4j"
        # graph_store.llm MUST be explicit (never rely on mem0ai's openai default)
        assert "llm" in config_dict["graph_store"]
        assert config_dict["graph_store"]["llm"]["provider"] == "anthropic"

    def test_graph_disabled(self):
        """Graph store omitted when MEM0_ENABLE_GRAPH=false."""
        env = {"MEM0_ENABLE_GRAPH": "false"}
        config_dict, *_ = self._build_with_env(env)
        assert "graph_store" not in config_dict

    def test_explicit_embedder_provider(self):
        """Embedder provider is always explicit (never default to openai)."""
        config_dict, *_ = self._build_with_env({})
        assert config_dict["embedder"]["provider"] == "ollama"

    def test_provider_info_structure(self):
        """Provider info tuple has the expected structure."""
        _, provider_info, *_ = self._build_with_env({})

        assert provider_info["name"] == "anthropic"
        assert "AnthropicOATLLM" in provider_info["class_path"]
        assert "AnthropicOATConfig" in provider_info["config_class_path"]

    def test_qdrant_optional_fields(self):
        """Optional Qdrant fields only included when env vars set."""
        config_dict, *_ = self._build_with_env({})
        assert "api_key" not in config_dict["vector_store"]["config"]

        env = {"MEM0_QDRANT_API_KEY": "test-key"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["vector_store"]["config"]["api_key"] == "test-key"

    def test_graph_llm_ollama(self):
        """Graph LLM can be set to ollama for quota savings."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "ollama",
            "MEM0_GRAPH_LLM_MODEL": "qwen3:14b",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "ollama"
        assert graph_llm["config"]["model"] == "qwen3:14b"

    def test_graph_llm_gemini(self):
        """Graph LLM can be set to gemini with API key."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini",
            "GOOGLE_API_KEY": "test-gemini-key",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "gemini"
        assert graph_llm["config"]["model"] == "gemini-2.5-flash-lite"
        assert graph_llm["config"]["api_key"] == "test-gemini-key"

    def test_graph_llm_gemini_model_override(self):
        """Gemini graph LLM model can be overridden via env var."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini",
            "MEM0_GRAPH_LLM_MODEL": "gemini-2.0-flash",
            "GOOGLE_API_KEY": "test-gemini-key",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["config"]["model"] == "gemini-2.0-flash"

    def test_graph_llm_gemini_no_api_key(self):
        """Gemini graph LLM config produced even without GOOGLE_API_KEY."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini",
        }
        config_dict, *_ = self._build_with_env(env)

        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "gemini"
        assert graph_llm["config"]["model"] == "gemini-2.5-flash-lite"
        assert "api_key" not in graph_llm["config"]

    def test_graph_llm_gemini_split_defaults(self):
        """Split-model config: config uses 'gemini' for validation, split_config returned separately."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-gemini-key",
        }
        config_dict, _, split_config = self._build_with_env(env)

        # Config dict uses "gemini" for pydantic validation
        graph_llm = config_dict["graph_store"]["llm"]
        assert graph_llm["provider"] == "gemini"
        assert graph_llm["config"]["model"] == "gemini-2.5-flash-lite"
        assert graph_llm["config"]["api_key"] == "test-gemini-key"

        # Split config returned separately for post-creation LLM swap
        assert split_config is not None
        assert split_config["extraction_provider"] == "gemini"
        assert split_config["extraction_model"] == "gemini-2.5-flash-lite"
        assert split_config["extraction_api_key"] == "test-gemini-key"
        assert split_config["contradiction_provider"] == "anthropic"
        assert split_config["contradiction_model"] == "claude-opus-4-6"
        assert split_config["contradiction_api_key"] == "sk-test-token"

    def test_graph_llm_gemini_split_custom_contradiction(self):
        """Split-model with custom contradiction provider/model."""
        env = {
            "MEM0_ENABLE_GRAPH": "true",
            "MEM0_GRAPH_LLM_PROVIDER": "gemini_split",
            "GOOGLE_API_KEY": "test-gemini-key",
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER": "ollama",
            "MEM0_GRAPH_CONTRADICTION_LLM_MODEL": "qwen3:14b",
        }
        config_dict, _, split_config = self._build_with_env(env)

        assert split_config is not None
        assert split_config["contradiction_provider"] == "ollama"
        assert split_config["contradiction_model"] == "qwen3:14b"
        assert "contradiction_api_key" not in split_config
        assert "contradiction_ollama_base_url" in split_config

    def test_no_split_config_for_regular_providers(self):
        """split_config is None when not using gemini_split."""
        env = {"MEM0_ENABLE_GRAPH": "true"}
        _, _, split_config = self._build_with_env(env)
        assert split_config is None

        env = {"MEM0_ENABLE_GRAPH": "true", "MEM0_GRAPH_LLM_PROVIDER": "gemini"}
        _, _, split_config = self._build_with_env(env)
        assert split_config is None
