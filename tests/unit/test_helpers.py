"""Tests for helpers.py — error wrapper, call_with_graph, bulk delete, user_id."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted.helpers import (
    _mem0_call,
    call_with_graph,
    get_default_user_id,
    safe_bulk_delete,
)


class TestMem0Call:
    def test_success_returns_json(self):
        result = _mem0_call(lambda: {"status": "ok"})
        parsed = json.loads(result)
        assert parsed == {"status": "ok"}

    def test_memory_error_caught(self):
        """MemoryError subclass returns structured error JSON."""
        # Create a mock MemoryError-like exception
        class FakeMemoryError(Exception):
            pass

        FakeMemoryError.__name__ = "MemoryError"

        exc = FakeMemoryError("something failed")
        exc.error_code = "VALIDATION_ERROR"
        exc.details = "missing field"
        exc.suggestion = "add user_id"

        # Patch the MRO check
        def _raise():
            raise exc

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_generic_exception_caught(self):
        """Generic Exception returns type name and detail."""
        def _raise():
            raise ValueError("bad input")

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert parsed["error"] == "ValueError"
        assert parsed["detail"] == "bad input"

    def test_ensure_ascii_false(self):
        """Non-ASCII characters preserved in output."""
        result = _mem0_call(lambda: {"text": "Alice prefiere TypeScript"})
        assert "prefiere" in result


class TestCallWithGraph:
    def test_sets_enable_graph_true(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is True
            return "ok"

        result = call_with_graph(memory, True, False, _check)
        assert result == "ok"

    def test_sets_enable_graph_false(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is False
            return "ok"

        result = call_with_graph(memory, False, True, _check)
        assert result == "ok"

    def test_uses_default_when_none(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is True
            return "ok"

        result = call_with_graph(memory, None, True, _check)
        assert result == "ok"

    def test_graph_none_forces_false(self):
        """If memory.graph is None, enable_graph stays False regardless."""
        memory = MagicMock()
        memory.graph = None

        def _check():
            assert memory.enable_graph is False
            return "ok"

        result = call_with_graph(memory, True, True, _check)
        assert result == "ok"


class TestSafeBulkDelete:
    def test_iterates_and_deletes(self):
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None

        # Mock vector_store.list returning items with .id
        item1 = MagicMock()
        item1.id = "id-1"
        item2 = MagicMock()
        item2.id = "id-2"
        memory.vector_store.list.return_value = [item1, item2]

        count = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert count == 2
        assert memory.delete.call_count == 2
        memory.delete.assert_any_call("id-1")
        memory.delete.assert_any_call("id-2")

    def test_graph_cleanup_when_enabled(self):
        memory = MagicMock()
        memory.enable_graph = True
        memory.graph = MagicMock()
        memory.vector_store.list.return_value = []

        safe_bulk_delete(memory, {"user_id": "testuser"})

        memory.graph.delete_all.assert_called_once_with({"user_id": "testuser"})

    def test_no_graph_cleanup_when_disabled(self):
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None
        memory.vector_store.list.return_value = []

        safe_bulk_delete(memory, {"user_id": "testuser"})

        # graph.delete_all should not be called when graph is disabled


class TestGetDefaultUserId:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        assert get_default_user_id() == "user"

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("MEM0_USER_ID", "bob")
        assert get_default_user_id() == "bob"
