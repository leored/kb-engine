"""Indexing service."""

from uuid import UUID

from kb_engine.core.exceptions import DocumentNotFoundError
from kb_engine.core.models.document import Document
from kb_engine.core.models.repository import RepositoryConfig
from kb_engine.core.models.search import SearchFilters
from kb_engine.pipelines.indexation import IndexationPipeline


class IndexingService:
    """Service for document indexing operations."""

    def __init__(self, pipeline: IndexationPipeline) -> None:
        self._pipeline = pipeline

    async def index_document(
        self,
        title: str,
        content: str,
        source_path: str | None = None,
        external_id: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Document:
        """Index a new document."""
        document = Document(
            title=title,
            content=content,
            source_path=source_path,
            external_id=external_id,
            domain=domain,
            tags=tags or [],
            metadata=metadata or {},
        )
        return await self._pipeline.index_document(document)

    async def reindex_document(self, document_id: UUID) -> Document:
        """Reindex an existing document."""
        document = await self._pipeline._traceability.get_document(document_id)
        if not document:
            raise DocumentNotFoundError(
                f"Document not found: {document_id}",
                details={"document_id": str(document_id)},
            )
        return await self._pipeline.reindex_document(document)

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document."""
        document = await self._pipeline._traceability.get_document(document_id)
        if not document:
            raise DocumentNotFoundError(
                f"Document not found: {document_id}",
                details={"document_id": str(document_id)},
            )
        return await self._pipeline.delete_document(document)

    async def get_document(self, document_id: UUID) -> Document:
        """Get a document by ID."""
        document = await self._pipeline._traceability.get_document(document_id)
        if not document:
            raise DocumentNotFoundError(
                f"Document not found: {document_id}",
                details={"document_id": str(document_id)},
            )
        return document

    async def list_documents(
        self,
        filters: SearchFilters | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents with optional filters."""
        return await self._pipeline._traceability.list_documents(
            filters=filters,
            limit=limit,
            offset=offset,
        )

    async def index_repository(self, repo_config: RepositoryConfig) -> list[Document]:
        """Index all matching files from a Git repository."""
        return await self._pipeline.index_repository(repo_config)

    async def sync_repository(self, repo_config: RepositoryConfig, since_commit: str) -> dict:
        """Incrementally sync a repository."""
        return await self._pipeline.sync_repository(repo_config, since_commit)
