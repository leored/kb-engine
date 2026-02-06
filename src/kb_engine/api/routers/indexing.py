"""Indexing API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from kb_engine.api.dependencies import IndexingServiceDep
from kb_engine.core.exceptions import DocumentNotFoundError
from kb_engine.core.models.document import Document, DocumentStatus

router = APIRouter(prefix="/indexing")


# --- Request/Response models ---

class IndexDocumentRequest(BaseModel):
    """Request model for indexing a document."""

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source_path: str | None = Field(default=None, max_length=1000)
    external_id: str | None = Field(default=None, max_length=255)
    domain: str | None = Field(default=None, max_length=100)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Response model for document operations."""

    id: UUID
    title: str
    status: DocumentStatus
    source_path: str | None
    external_id: str | None
    domain: str | None
    tags: list[str]
    repo_name: str | None = None
    relative_path: str | None = None

    class Config:
        from_attributes = True


class RegisterRepositoryRequest(BaseModel):
    """Request to register a Git repository for indexing."""

    name: str = Field(..., min_length=1, max_length=255)
    local_path: str = Field(..., min_length=1)
    remote_url: str | None = None
    branch: str = "main"
    include_patterns: list[str] = Field(default_factory=lambda: ["**/*.md"])
    exclude_patterns: list[str] = Field(default_factory=list)
    base_url_template: str | None = None


class SyncRepositoryRequest(BaseModel):
    """Request to sync a repository from a specific commit."""

    since_commit: str = Field(..., min_length=7, max_length=40)


class RepositoryIndexResult(BaseModel):
    """Result of a repository indexing operation."""

    repo_name: str
    documents_indexed: int
    status: str = "completed"


class RepositorySyncResult(BaseModel):
    """Result of a repository sync operation."""

    repo_name: str
    commit: str
    indexed: int
    deleted: int
    skipped: int


# --- Document endpoints ---

@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def index_document(
    request: IndexDocumentRequest,
    service: IndexingServiceDep,
) -> Document:
    """Index a new document."""
    return await service.index_document(
        title=request.title,
        content=request.content,
        source_path=request.source_path,
        external_id=request.external_id,
        domain=request.domain,
        tags=request.tags,
        metadata=request.metadata,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    service: IndexingServiceDep,
) -> Document:
    """Get a document by ID."""
    try:
        return await service.get_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )


@router.post("/documents/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: UUID,
    service: IndexingServiceDep,
) -> Document:
    """Reindex an existing document."""
    try:
        return await service.reindex_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    service: IndexingServiceDep,
) -> None:
    """Delete a document and all its indexed data."""
    try:
        await service.delete_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    service: IndexingServiceDep,
    limit: int = 100,
    offset: int = 0,
    domain: str | None = None,
) -> list[Document]:
    """List indexed documents."""
    from kb_engine.core.models.search import SearchFilters

    filters = None
    if domain:
        filters = SearchFilters(domains=[domain])

    return await service.list_documents(
        filters=filters,
        limit=limit,
        offset=offset,
    )


# --- Repository endpoints ---

@router.post(
    "/repositories",
    response_model=RepositoryIndexResult,
    status_code=status.HTTP_201_CREATED,
)
async def register_and_index_repository(
    request: RegisterRepositoryRequest,
    service: IndexingServiceDep,
) -> RepositoryIndexResult:
    """Register a Git repository and index all matching files."""
    from kb_engine.core.models.repository import RepositoryConfig

    config = RepositoryConfig(
        name=request.name,
        local_path=request.local_path,
        remote_url=request.remote_url,
        branch=request.branch,
        include_patterns=request.include_patterns,
        exclude_patterns=request.exclude_patterns,
        base_url_template=request.base_url_template,
    )

    documents = await service.index_repository(config)
    return RepositoryIndexResult(
        repo_name=request.name,
        documents_indexed=len(documents),
    )


@router.post(
    "/repositories/{name}/sync",
    response_model=RepositorySyncResult,
)
async def sync_repository(
    name: str,
    request: SyncRepositoryRequest,
    service: IndexingServiceDep,
) -> RepositorySyncResult:
    """Incrementally sync a repository (only changed files)."""
    from kb_engine.core.models.repository import RepositoryConfig

    # For now, the caller must provide the full config
    # In a future version, we'd store repo configs in the DB
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Repository sync via API requires stored repo config. Use the CLI instead.",
    )


@router.post(
    "/repositories/{name}/reindex",
    response_model=RepositoryIndexResult,
)
async def reindex_repository(
    name: str,
    service: IndexingServiceDep,
) -> RepositoryIndexResult:
    """Full reindex of a repository."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Repository reindex via API requires stored repo config. Use the CLI instead.",
    )
