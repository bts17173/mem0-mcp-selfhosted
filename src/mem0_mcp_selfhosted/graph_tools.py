"""Direct Neo4j graph tools.

Provides search_graph and get_entity via Cypher queries with a
lazily-initialized Neo4j driver separate from mem0ai's graph store.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized Neo4j driver — created on first tool call, not at startup.
_driver = None


def _get_driver():
    """Get or create the Neo4j driver (lazy initialization)."""
    global _driver
    if _driver is not None:
        return _driver

    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None

    url = os.environ.get("MEM0_NEO4J_URL", "bolt://127.0.0.1:7687")
    user = os.environ.get("MEM0_NEO4J_USER", "neo4j")
    password = os.environ.get("MEM0_NEO4J_PASSWORD", "mem0graph")
    database = os.environ.get("MEM0_NEO4J_DATABASE")

    try:
        _driver = GraphDatabase.driver(url, auth=(user, password))
        if database:
            _driver._default_database = database
        return _driver
    except Exception as exc:
        logger.error("Failed to create Neo4j driver: %s", exc)
        return None


def _run_query(query: str, params: dict[str, Any] | None = None) -> list[dict]:
    """Run a Cypher query and return results as list of dicts."""
    driver = _get_driver()
    if driver is None:
        raise RuntimeError("Neo4j driver not available. Check Neo4j connection settings.")

    database = os.environ.get("MEM0_NEO4J_DATABASE")
    with driver.session(database=database) if database else driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def search_graph(query: str) -> str:
    """Search entities by name substring in Neo4j.

    Returns matching entities and their outgoing relationships,
    formatted as 'source --[relationship]--> target', limited to 25.
    """
    try:
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($search_term)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.name AS entity,
               labels(n) AS labels,
               type(r) AS relationship,
               m.name AS related_entity
        LIMIT 25
        """
        records = _run_query(cypher, {"search_term": query})

        entities: list[dict[str, Any]] = []
        for record in records:
            entry: dict[str, Any] = {
                "entity": record["entity"],
                "labels": record.get("labels", []),
            }
            if record.get("relationship") and record.get("related_entity"):
                entry["relationship"] = (
                    f"{record['entity']} --[{record['relationship']}]--> {record['related_entity']}"
                )
            entities.append(entry)

        return json.dumps({"entities": entities}, ensure_ascii=False)

    except RuntimeError as exc:
        return json.dumps(
            {"error": "graph_unavailable", "detail": str(exc)},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("search_graph failed: %s", exc)
        return json.dumps(
            {"error": type(exc).__name__, "detail": str(exc)},
            ensure_ascii=False,
        )


def get_entity(name: str) -> str:
    """Get all relationships for a specific entity (bidirectional).

    Returns the full entity profile with outgoing and incoming connections.
    Returns empty result set (not error) if entity not found.
    """
    try:
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($name)
        OPTIONAL MATCH (n)-[r_out]->(target)
        OPTIONAL MATCH (source)-[r_in]->(n)
        RETURN n.name AS entity,
               labels(n) AS labels,
               collect(DISTINCT {
                   direction: 'outgoing',
                   relationship: type(r_out),
                   entity: target.name
               }) AS outgoing,
               collect(DISTINCT {
                   direction: 'incoming',
                   relationship: type(r_in),
                   entity: source.name
               }) AS incoming
        """
        records = _run_query(cypher, {"name": name})

        if not records:
            return json.dumps({"entity": name, "relationships": []}, ensure_ascii=False)

        record = records[0]
        relationships = []

        for rel in record.get("outgoing", []):
            if rel.get("relationship") and rel.get("entity"):
                relationships.append({
                    "direction": "outgoing",
                    "relationship": rel["relationship"],
                    "target": rel["entity"],
                })
        for rel in record.get("incoming", []):
            if rel.get("relationship") and rel.get("entity"):
                relationships.append({
                    "direction": "incoming",
                    "relationship": rel["relationship"],
                    "source": rel["entity"],
                })

        return json.dumps(
            {
                "entity": record["entity"],
                "labels": record.get("labels", []),
                "relationships": relationships,
            },
            ensure_ascii=False,
        )

    except RuntimeError as exc:
        return json.dumps(
            {"error": "graph_unavailable", "detail": str(exc)},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("get_entity failed: %s", exc)
        return json.dumps(
            {"error": type(exc).__name__, "detail": str(exc)},
            ensure_ascii=False,
        )
