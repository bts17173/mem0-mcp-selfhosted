"""Tests for auth.py — token resolution fallback chain and OAT detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mem0_mcp_selfhosted.auth import _CREDENTIALS_PATH, is_oat_token, resolve_token


class TestIsOatToken:
    def test_oat_token_detected(self):
        assert is_oat_token("sk-ant-oat01-abc123") is True

    def test_api_key_not_oat(self):
        assert is_oat_token("sk-ant-api03-xyz789") is False

    def test_empty_string(self):
        assert is_oat_token("") is False

    def test_partial_match(self):
        assert is_oat_token("sk-ant-oat") is True


class TestResolveToken:
    """Test the prioritized fallback chain."""

    def test_priority_1_env_var(self, monkeypatch):
        """MEM0_ANTHROPIC_TOKEN takes highest priority."""
        monkeypatch.setenv("MEM0_ANTHROPIC_TOKEN", "sk-ant-oat01-from-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-use")
        assert resolve_token() == "sk-ant-oat01-from-env"

    def test_priority_2_credentials_file(self, monkeypatch, tmp_path):
        """Falls back to credentials file when env var not set."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        creds = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-from-creds"}}
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps(creds))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert resolve_token() == "sk-ant-oat01-from-creds"

    def test_priority_3_api_key(self, monkeypatch):
        """Falls back to ANTHROPIC_API_KEY when others unavailable."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fallback")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert resolve_token() == "sk-ant-api03-fallback"

    def test_no_token_returns_none(self, monkeypatch):
        """Returns None when no auth is available."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            assert resolve_token() is None

    def test_missing_credentials_file_silent(self, monkeypatch):
        """Missing credentials file doesn't log error, just skips."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", Path("/nonexistent")):
            result = resolve_token()
            assert result is None

    def test_malformed_json_credentials(self, monkeypatch, tmp_path):
        """Malformed JSON in credentials file warns and continues."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fallback")

        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text("{bad json")

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert resolve_token() == "sk-ant-api03-fallback"

    def test_missing_access_token_key(self, monkeypatch, tmp_path):
        """Missing accessToken key warns and continues."""
        monkeypatch.delenv("MEM0_ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fallback")

        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps({"claudeAiOauth": {"noToken": True}}))

        with patch("mem0_mcp_selfhosted.auth._CREDENTIALS_PATH", creds_file):
            assert resolve_token() == "sk-ant-api03-fallback"
