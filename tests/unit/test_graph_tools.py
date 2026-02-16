"""Tests for graph_tools.py — lazy driver, error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import mem0_mcp_selfhosted.graph_tools as gt


class TestLazyDriverInit:
    def setup_method(self):
        """Reset the global driver before each test."""
        gt._driver = None

    def test_driver_created_on_first_use(self):
        """Driver is lazily created on first _get_driver() call."""
        mock_gdb = MagicMock()
        mock_driver = MagicMock()
        mock_gdb.GraphDatabase.driver.return_value = mock_driver

        # neo4j is imported lazily inside _get_driver, so patch sys.modules
        with patch.dict("sys.modules", {"neo4j": mock_gdb}):
            gt._driver = None
            driver = gt._get_driver()
            assert driver is mock_driver
            mock_gdb.GraphDatabase.driver.assert_called_once()

    def test_driver_reused_on_subsequent_calls(self):
        """Once created, driver is reused."""
        mock_driver = MagicMock()
        gt._driver = mock_driver
        assert gt._get_driver() is mock_driver


class TestSearchGraph:
    def test_neo4j_unavailable_returns_error(self):
        """Returns structured error when Neo4j is unreachable."""
        gt._driver = None
        with patch("mem0_mcp_selfhosted.graph_tools._get_driver", return_value=None):
            result = json.loads(gt.search_graph("Alice"))
            assert result["error"] == "graph_unavailable"

    def test_successful_search(self):
        """Successful Cypher query returns formatted entities."""
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "entity": "Alice",
            "labels": ["Person"],
            "relationship": "PREFERS",
            "related_entity": "TypeScript",
        }
        mock_session.run.return_value = [mock_record]
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.search_graph("Alice"))

        assert "entities" in result
        assert len(result["entities"]) == 1
        assert "Alice --[PREFERS]--> TypeScript" in result["entities"][0]["relationship"]


class TestGetEntity:
    def test_entity_not_found_returns_empty(self):
        """Returns empty result set (not error) when entity not found."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        gt._driver = mock_driver
        with patch.dict("os.environ", {}, clear=False):
            result = json.loads(gt.get_entity("NonExistent"))

        assert result["entity"] == "NonExistent"
        assert result["relationships"] == []

    def test_neo4j_unavailable_returns_error(self):
        """Returns structured error when Neo4j is unreachable."""
        gt._driver = None
        with patch("mem0_mcp_selfhosted.graph_tools._get_driver", return_value=None):
            result = json.loads(gt.get_entity("Alice"))
            assert result["error"] == "graph_unavailable"
