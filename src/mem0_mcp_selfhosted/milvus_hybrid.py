"""Hybrid (Dense + BM25) MilvusDB for mem0-mcp-selfhosted.

Subclasses mem0's MilvusDB to add:
- BM25 full-text search via sparse_vector field
- Hybrid search with RRF (Reciprocal Rank Fusion) ranking
- Chinese analyzer for BM25 tokenization
"""

from __future__ import annotations

import logging
from typing import Optional

from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from mem0.vector_stores.milvus import MilvusDB, OutputData

logger = logging.getLogger(__name__)


class HybridMilvusDB(MilvusDB):
    """MilvusDB with BM25 hybrid search support."""

    def __init__(self, **kwargs):
        # Extract hybrid-specific config before calling super
        self.enable_hybrid = kwargs.pop("enable_hybrid", True)
        self.analyzer_type = kwargs.pop("analyzer_type", "chinese")
        super().__init__(**kwargs)

    def create_col(self, collection_name, vector_size, metric_type=None):
        """Create collection with dense + sparse (BM25) vector fields."""
        if metric_type is None:
            from mem0.configs.vector_stores.milvus import MetricType
            metric_type = MetricType.COSINE

        if self.client.has_collection(collection_name):
            logger.info("Collection %s already exists. Skipping creation.", collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
            FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        if self.enable_hybrid:
            fields.extend([
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                    analyzer_params={"type": self.analyzer_type},
                ),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, is_function_output=True),
            ])

        schema = CollectionSchema(fields, enable_dynamic_field=True)

        if self.enable_hybrid:
            bm25_fn = Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
            )
            schema.add_function(bm25_fn)

        # Index params for both dense and sparse vectors
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vectors",
            metric_type=metric_type,
            index_type="AUTOINDEX",
            index_name="vector_index",
        )
        if self.enable_hybrid:
            index_params.add_index(
                field_name="sparse_vector",
                metric_type="IP",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_index",
            )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created hybrid collection %s (hybrid=%s)", collection_name, self.enable_hybrid)

    def insert(self, ids, vectors, payloads, **kwargs):
        """Insert with text field extracted from payload for BM25."""
        data = []
        for idx, embedding, metadata in zip(ids, vectors, payloads):
            record = {"id": idx, "vectors": embedding, "metadata": metadata}
            if self.enable_hybrid:
                # Extract text from payload's "data" field (where mem0 stores the memory text)
                record["text"] = metadata.get("data", "") if isinstance(metadata, dict) else ""
            data.append(record)
        self.client.insert(collection_name=self.collection_name, data=data, **kwargs)

    def search(self, query: str, vectors: list, limit: int = 5, filters: Optional[dict] = None) -> list:
        """Hybrid search: dense vector + BM25 sparse, fused with RRF."""
        query_filter = self._create_filter(filters) if filters else None

        if not self.enable_hybrid:
            # Fallback to pure dense search
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[vectors],
                limit=limit,
                filter=query_filter,
                output_fields=["*"],
            )
            return self._parse_output(data=hits[0])

        # Dense search request
        dense_params = {
            "data": [vectors],
            "anns_field": "vectors",
            "param": {"metric_type": "COSINE"},
            "limit": limit,
        }
        if query_filter:
            dense_params["expr"] = query_filter
        dense_req = AnnSearchRequest(**dense_params)

        # Sparse (BM25) search request
        sparse_params = {
            "data": [query],
            "anns_field": "sparse_vector",
            "param": {"metric_type": "BM25"},
            "limit": limit,
        }
        if query_filter:
            sparse_params["expr"] = query_filter
        sparse_req = AnnSearchRequest(**sparse_params)

        # Hybrid search with RRF ranking
        hits = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["*"],
        )

        # Parse hybrid results into mem0 OutputData format
        results = []
        for hit in hits[0]:
            entity = hit.get("entity", hit) if isinstance(hit, dict) else hit
            metadata = entity.get("metadata", {}) if isinstance(entity, dict) else getattr(entity, "entity", {}).get("metadata", {})
            results.append(
                OutputData(
                    id=hit["id"] if isinstance(hit, dict) else hit.id,
                    score=hit["distance"] if isinstance(hit, dict) else hit.distance,
                    payload=metadata,
                )
            )
        return results
