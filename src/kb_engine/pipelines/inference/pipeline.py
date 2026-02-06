"""Retrieval pipeline - returns document references with URLs."""

import time

import structlog

from kb_engine.core.models.search import (
    DocumentReference,
    RetrievalMode,
    RetrievalResponse,
    SearchFilters,
)
from kb_engine.embedding import EmbeddingConfig, EmbeddingProviderFactory
from kb_engine.git.url_resolver import URLResolver
from kb_engine.utils.markdown import extract_snippet, heading_path_to_anchor

logger = structlog.get_logger(__name__)


class RetrievalPipeline:
    """Pipeline for processing retrieval queries.

    Returns DocumentReference objects with full URLs (including #anchors)
    instead of raw document content. This allows external agents to
    read the source documents directly.
    """

    def __init__(
        self,
        traceability_repo,
        vector_repo,
        graph_repo=None,
        url_resolver: URLResolver | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self._traceability = traceability_repo
        self._vector = vector_repo
        self._graph = graph_repo
        self._url_resolver = url_resolver

        self._embedding_provider = EmbeddingProviderFactory(embedding_config).create_provider()

    async def search(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.VECTOR,
        filters: SearchFilters | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Execute a retrieval query, returning document references."""
        start_time = time.time()

        references: list[DocumentReference] = []

        if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID):
            vector_refs = await self._vector_search(
                query, filters, limit, score_threshold
            )
            references.extend(vector_refs)

        if mode in (RetrievalMode.GRAPH, RetrievalMode.HYBRID):
            graph_refs = await self._graph_search(query, filters, limit)
            references.extend(graph_refs)

        # Deduplicate by URL if hybrid
        if mode == RetrievalMode.HYBRID:
            references = self._deduplicate_references(references, limit)

        # Sort by score descending
        references.sort(key=lambda r: r.score, reverse=True)
        references = references[:limit]

        processing_time = (time.time() - start_time) * 1000

        return RetrievalResponse(
            query=query,
            references=references,
            total_count=len(references),
            processing_time_ms=processing_time,
        )

    async def _vector_search(
        self,
        query: str,
        filters: SearchFilters | None,
        limit: int,
        score_threshold: float | None,
    ) -> list[DocumentReference]:
        """Perform vector similarity search and resolve to references."""
        query_vector = await self._embedding_provider.embed_text(query)

        chunk_scores = await self._vector.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )

        references = []
        for chunk_id, score in chunk_scores:
            chunk = await self._traceability.get_chunk(chunk_id)
            if not chunk:
                continue

            document = await self._traceability.get_document(chunk.document_id)
            if not document:
                continue

            # Resolve URL
            anchor = chunk.section_anchor or heading_path_to_anchor(chunk.heading_path)
            if self._url_resolver and document.relative_path:
                url = self._url_resolver.resolve(document.relative_path, anchor)
            elif document.source_path:
                url = f"file://{document.source_path}"
                if anchor:
                    url += f"#{anchor}"
            else:
                url = f"doc://{document.id}"

            # Build section title from heading path
            section_title = chunk.heading_path[-1] if chunk.heading_path else None

            references.append(
                DocumentReference(
                    url=url,
                    document_path=document.relative_path or document.source_path or "",
                    section_anchor=anchor,
                    title=document.title,
                    section_title=section_title,
                    score=score,
                    snippet=extract_snippet(chunk.content),
                    domain=document.domain,
                    tags=document.tags,
                    chunk_type=chunk.chunk_type.value,
                    metadata=chunk.metadata,
                    retrieval_mode=RetrievalMode.VECTOR,
                )
            )

        return references

    async def _graph_search(
        self,
        query: str,
        filters: SearchFilters | None,
        limit: int,
    ) -> list[DocumentReference]:
        """Graph-based search (placeholder for future implementation)."""
        return []

    def _deduplicate_references(
        self,
        references: list[DocumentReference],
        limit: int,
    ) -> list[DocumentReference]:
        """Deduplicate references using Reciprocal Rank Fusion."""
        url_scores: dict[str, tuple[DocumentReference, float]] = {}
        k = 60  # RRF constant

        for rank, ref in enumerate(references):
            rrf_score = 1.0 / (k + rank + 1)
            if ref.url in url_scores:
                existing_ref, existing_score = url_scores[ref.url]
                url_scores[ref.url] = (existing_ref, existing_score + rrf_score)
            else:
                url_scores[ref.url] = (ref, rrf_score)

        merged = []
        for ref, rrf_score in url_scores.values():
            ref.score = rrf_score
            ref.retrieval_mode = RetrievalMode.HYBRID
            merged.append(ref)

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:limit]
