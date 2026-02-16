"""Contract tests: mem0ai internal API stability.

These tests validate assumptions about mem0ai internals that our code depends on.
If these fail after a mem0ai upgrade, our code needs updating.

NOTE: These tests require mem0ai to be installed. They test the real package,
not mocks. Skip with `pytest -m "not contract"` if deps unavailable.
"""

from __future__ import annotations

import pytest

# Mark all tests in this module as contract tests
pytestmark = pytest.mark.contract


class TestVectorStoreClientAccess:
    """Test that memory.vector_store.client is a public, stable attribute."""

    def test_qdrant_class_has_client_attribute(self):
        """The Qdrant vector store class exposes .client as a public attribute."""
        try:
            from mem0.vector_stores.qdrant import Qdrant
        except ImportError:
            pytest.skip("mem0ai not installed")

        # Verify 'client' is in the class (not a private _client)
        assert hasattr(Qdrant, "__init__"), "Qdrant class must have __init__"
        # Check the source to verify client is assigned (not _client)
        import inspect

        source = inspect.getsource(Qdrant.__init__)
        assert "self.client" in source, (
            "INVARIANT BROKEN: Qdrant.__init__ must assign self.client. "
            "Our code accesses memory.vector_store.client directly."
        )

    def test_qdrant_class_has_collection_name(self):
        """The Qdrant vector store class exposes .collection_name."""
        try:
            from mem0.vector_stores.qdrant import Qdrant
        except ImportError:
            pytest.skip("mem0ai not installed")

        import inspect

        source = inspect.getsource(Qdrant.__init__)
        assert "self.collection_name" in source, (
            "INVARIANT BROKEN: Qdrant.__init__ must assign self.collection_name. "
            "Our code accesses memory.vector_store.collection_name."
        )


class TestLlmFactoryRegistration:
    """Test LlmFactory.register_provider() behavior."""

    def test_register_provider_exists(self):
        """LlmFactory has a register_provider classmethod."""
        try:
            from mem0.utils.factory import LlmFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        assert hasattr(LlmFactory, "register_provider"), (
            "INVARIANT BROKEN: LlmFactory must have register_provider classmethod."
        )

    def test_register_provider_is_idempotent(self):
        """Calling register_provider twice with same name doesn't error."""
        try:
            from mem0.utils.factory import LlmFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        # Register once
        LlmFactory.register_provider(
            name="test_idempotent",
            class_path="mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            config_class=None,
        )
        # Register again — should not raise
        LlmFactory.register_provider(
            name="test_idempotent",
            class_path="mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            config_class=None,
        )

    def test_registration_persists_across_calls(self):
        """Registered provider persists in factory after registration."""
        try:
            from mem0.utils.factory import LlmFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        LlmFactory.register_provider(
            name="test_persist",
            class_path="mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            config_class=None,
        )

        # Verify the provider is in the factory's registry
        # The factory uses a class-level dict, so it should persist
        provider_map = getattr(LlmFactory, "provider_to_class", None)
        if provider_map is not None:
            assert "test_persist" in provider_map, (
                "INVARIANT BROKEN: Registered provider must persist in LlmFactory."
            )
