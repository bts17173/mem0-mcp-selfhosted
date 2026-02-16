"""Integration tests: safe_bulk_delete and list_entities_facet against real Qdrant.

These validate custom code paths that bypass mem0ai's memory.delete_all().
"""

from __future__ import annotations

import pytest

from mem0_mcp_selfhosted.helpers import list_entities_facet, safe_bulk_delete

pytestmark = pytest.mark.integration


class TestBulkOperations:
    def test_safe_bulk_delete(self, memory_instance, test_user_id):
        """Add 3 memories, bulk-delete them, verify all removed."""
        for i in range(3):
            memory_instance.add(
                [{"role": "user", "content": f"Bulk delete test fact number {i + 1}"}],
                user_id=test_user_id,
            )

        count = safe_bulk_delete(memory_instance, {"user_id": test_user_id})

        assert count == 3

        remaining = memory_instance.get_all(user_id=test_user_id)
        assert len(remaining.get("results", [])) == 0

    def test_list_entities_facet(self, memory_instance):
        """Add memories with distinct user_ids, verify Facet API returns them."""
        user_a = "inttest-facet-user-a"
        user_b = "inttest-facet-user-b"

        # Add 2 memories for user_a
        memory_instance.add(
            [{"role": "user", "content": "Facet test: user A fact one"}],
            user_id=user_a,
        )
        memory_instance.add(
            [{"role": "user", "content": "Facet test: user A fact two"}],
            user_id=user_a,
        )
        # Add 1 memory for user_b
        memory_instance.add(
            [{"role": "user", "content": "Facet test: user B fact one"}],
            user_id=user_b,
        )

        result = list_entities_facet(memory_instance)

        assert "users" in result
        user_map = {u["value"]: u["count"] for u in result["users"]}
        assert user_map.get(user_a, 0) >= 2
        assert user_map.get(user_b, 0) >= 1
