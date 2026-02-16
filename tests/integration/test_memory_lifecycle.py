"""Integration tests: full memory round-trip against live infrastructure.

Each test is independent — creates its own data via unique test_user_id.
All use infer=True (default) to exercise the full AnthropicOATLLM provider.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestMemoryLifecycle:
    def test_add_memory(self, memory_instance, test_user_id):
        """Add a memory and verify the result structure."""
        result = memory_instance.add(
            [{"role": "user", "content": "I prefer Rust for systems programming"}],
            user_id=test_user_id,
        )

        assert "results" in result
        assert len(result["results"]) >= 1
        assert "id" in result["results"][0]

    def test_search_finds_added_memory(self, memory_instance, test_user_id):
        """Add a distinctive fact, then search semantically for it."""
        memory_instance.add(
            [{"role": "user", "content": "I prefer Hatch for Python packaging"}],
            user_id=test_user_id,
        )

        results = memory_instance.search(
            query="Python packaging tool preference",
            user_id=test_user_id,
        )

        assert "results" in results
        assert len(results["results"]) >= 1
        # At least one result should reference the distinctive fact
        memories_text = " ".join(
            r.get("memory", "") for r in results["results"]
        ).lower()
        assert "hatch" in memories_text or "python" in memories_text

    def test_get_memory_by_id(self, memory_instance, test_user_id):
        """Add a memory, then fetch it by UUID."""
        add_result = memory_instance.add(
            [{"role": "user", "content": "My favorite database is PostgreSQL"}],
            user_id=test_user_id,
        )
        memory_id = add_result["results"][0]["id"]

        get_result = memory_instance.get(memory_id)

        assert get_result is not None
        assert "memory" in get_result

    def test_get_all_memories(self, memory_instance, test_user_id):
        """Add a memory, then list all memories for the test user."""
        add_result = memory_instance.add(
            [{"role": "user", "content": "I use tmux as my terminal multiplexer"}],
            user_id=test_user_id,
        )
        memory_id = add_result["results"][0]["id"]

        all_result = memory_instance.get_all(user_id=test_user_id)

        assert "results" in all_result
        ids = [r["id"] for r in all_result["results"]]
        assert memory_id in ids

    def test_update_then_search(self, memory_instance, test_user_id):
        """Add a memory, update its text, then search for the new content."""
        add_result = memory_instance.add(
            [{"role": "user", "content": "I use VS Code as my editor"}],
            user_id=test_user_id,
        )
        memory_id = add_result["results"][0]["id"]

        memory_instance.update(memory_id, data="I use Neovim as my editor")

        results = memory_instance.search(
            query="text editor preference",
            user_id=test_user_id,
        )

        assert "results" in results
        assert len(results["results"]) >= 1
        memories_text = " ".join(
            r.get("memory", "") for r in results["results"]
        ).lower()
        assert "neovim" in memories_text

    def test_delete_removes_memory(self, memory_instance, test_user_id):
        """Add a memory, delete it, verify it's gone."""
        add_result = memory_instance.add(
            [{"role": "user", "content": "I prefer dark mode themes"}],
            user_id=test_user_id,
        )
        memory_id = add_result["results"][0]["id"]

        memory_instance.delete(memory_id)

        get_result = memory_instance.get(memory_id)
        # Deleted memory returns None or empty
        assert get_result is None or get_result == {}
