"""Retrieval service."""

from kb_engine.core.models.search import RetrievalMode, RetrievalResponse, SearchFilters
from kb_engine.pipelines.inference.pipeline import RetrievalPipeline


class RetrievalService:
    """Service for document retrieval operations.

    Returns DocumentReference objects with URLs instead of raw content.
    """

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        self._pipeline = pipeline

    async def search(
        self,
        query: str,
        mode: RetrievalMode | str = RetrievalMode.VECTOR,
        filters: SearchFilters | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Execute a retrieval query."""
        if isinstance(mode, str):
            mode = RetrievalMode(mode.lower())

        return await self._pipeline.search(
            query=query,
            mode=mode,
            filters=filters,
            limit=limit,
            score_threshold=score_threshold,
        )
