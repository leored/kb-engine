"""Retrieval API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from kb_engine.api.dependencies import RetrievalServiceDep
from kb_engine.core.models.search import RetrievalMode, RetrievalResponse, SearchFilters

router = APIRouter(prefix="/retrieval")


class RetrievalRequest(BaseModel):
    """Request model for retrieval endpoint."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    mode: RetrievalMode = Field(
        default=RetrievalMode.VECTOR, description="Retrieval mode"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Min score threshold"
    )
    filters: SearchFilters | None = Field(default=None, description="Search filters")


@router.post("/search", response_model=RetrievalResponse)
async def search(
    request: RetrievalRequest,
    service: RetrievalServiceDep,
) -> RetrievalResponse:
    """Search the knowledge base and return document references with URLs."""
    return await service.search(
        query=request.query,
        mode=request.mode,
        filters=request.filters,
        limit=request.limit,
        score_threshold=request.score_threshold,
    )


@router.get("/search", response_model=RetrievalResponse)
async def search_get(
    service: RetrievalServiceDep,
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    mode: RetrievalMode = RetrievalMode.VECTOR,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> RetrievalResponse:
    """Search the knowledge base (GET variant for simple queries)."""
    return await service.search(
        query=query,
        mode=mode,
        limit=limit,
    )
