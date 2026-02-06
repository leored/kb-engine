"""Main indexation pipeline."""

from datetime import datetime
from pathlib import Path

import structlog

from kb_engine.chunking import ChunkerFactory, ChunkingConfig
from kb_engine.core.exceptions import PipelineError
from kb_engine.core.models.document import Document, DocumentStatus
from kb_engine.core.models.graph import Node
from kb_engine.core.models.repository import RepositoryConfig
from kb_engine.embedding import EmbeddingConfig, EmbeddingProviderFactory
from kb_engine.extraction import ExtractionConfig, ExtractionPipelineFactory
from kb_engine.git.scanner import GitRepoScanner
from kb_engine.git.url_resolver import URLResolver
from kb_engine.utils.hashing import compute_content_hash
from kb_engine.utils.markdown import extract_frontmatter, heading_path_to_anchor

logger = structlog.get_logger(__name__)


class IndexationPipeline:
    """Pipeline for indexing documents into the knowledge base.

    Orchestrates the full indexation process:
    1. Parse and validate document
    2. Chunk content using semantic strategies
    3. Compute section anchors for each chunk
    4. Generate embeddings
    5. Extract entities and relationships (optional)
    6. Store in all repositories
    """

    def __init__(
        self,
        traceability_repo,
        vector_repo,
        graph_repo=None,
        url_resolver: URLResolver | None = None,
        chunking_config: ChunkingConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self._traceability = traceability_repo
        self._vector = vector_repo
        self._graph = graph_repo
        self._url_resolver = url_resolver

        # Initialize components
        self._chunker = ChunkerFactory(chunking_config)
        self._embedding_provider = EmbeddingProviderFactory(embedding_config).create_provider()
        self._extraction_pipeline = ExtractionPipelineFactory(extraction_config).create_pipeline()

    async def index_document(self, document: Document) -> Document:
        """Index a document through the full pipeline."""
        try:
            document.status = DocumentStatus.PROCESSING
            document.content_hash = compute_content_hash(document.content)

            # 1. Save document to traceability store
            document = await self._traceability.save_document(document)

            # 2. Chunk the document
            chunks = self._chunker.chunk_document(document)

            # 3. Compute section anchors from heading paths
            for chunk in chunks:
                chunk.section_anchor = heading_path_to_anchor(chunk.heading_path)

            # 4. Save chunks to traceability store
            chunks = await self._traceability.save_chunks(chunks)

            # 5. Generate embeddings
            embeddings = await self._embedding_provider.embed_chunks(chunks)

            # 6. Store embeddings in vector store
            await self._vector.upsert_embeddings(embeddings)

            # 7. Extract entities and store in graph (if graph repo available)
            if self._graph is not None:
                extraction_result = await self._extraction_pipeline.extract_document(
                    document, chunks
                )
                for node_data in extraction_result.nodes:
                    node = Node(
                        name=node_data.name,
                        node_type=node_data.node_type,
                        description=node_data.description,
                        source_document_id=document.id,
                        properties=node_data.properties,
                        confidence=node_data.confidence,
                        extraction_method=node_data.extraction_method,
                    )
                    await self._graph.create_node(node)

            # 8. Update document status
            document.status = DocumentStatus.INDEXED
            document.indexed_at = datetime.utcnow()
            document = await self._traceability.update_document(document)

            logger.info(
                "Document indexed",
                document_id=str(document.id),
                title=document.title,
                chunks=len(chunks),
            )
            return document

        except Exception as e:
            document.status = DocumentStatus.FAILED
            try:
                await self._traceability.update_document(document)
            except Exception:
                pass
            raise PipelineError(
                f"Failed to index document: {e}",
                details={"document_id": str(document.id)},
            ) from e

    async def reindex_document(self, document: Document) -> Document:
        """Reindex an existing document."""
        await self._vector.delete_by_document(document.id)
        if self._graph is not None:
            await self._graph.delete_by_document(document.id)
        await self._traceability.delete_chunks_by_document(document.id)
        return await self.index_document(document)

    async def delete_document(self, document: Document) -> bool:
        """Delete a document and all its indexed data."""
        await self._vector.delete_by_document(document.id)
        if self._graph is not None:
            await self._graph.delete_by_document(document.id)
        await self._traceability.delete_chunks_by_document(document.id)
        return await self._traceability.delete_document(document.id)

    async def index_repository(self, repo_config: RepositoryConfig) -> list[Document]:
        """Index all matching files from a Git repository."""
        scanner = GitRepoScanner(repo_config)
        if not scanner.is_git_repo():
            raise PipelineError(f"Not a git repository: {repo_config.local_path}")

        resolver = self._url_resolver or URLResolver(repo_config)
        commit = scanner.get_current_commit()
        remote_url = scanner.get_remote_url()
        files = scanner.scan_files()

        logger.info(
            "Indexing repository",
            repo=repo_config.name,
            files=len(files),
            commit=commit[:8],
        )

        documents = []
        for relative_path in files:
            try:
                content = scanner.read_file(relative_path)
                title = Path(relative_path).stem
                frontmatter, body = extract_frontmatter(content)

                doc = Document(
                    title=frontmatter.get("title", title),
                    content=content,
                    source_path=str(scanner.repo_path / relative_path),
                    external_id=f"{repo_config.name}:{relative_path}",
                    domain=frontmatter.get("domain"),
                    tags=frontmatter.get("tags", []),
                    metadata=frontmatter,
                    repo_name=repo_config.name,
                    relative_path=relative_path,
                    git_commit=commit,
                    git_remote_url=remote_url,
                )

                doc = await self.index_document(doc)
                documents.append(doc)
            except Exception as e:
                logger.error(
                    "Failed to index file",
                    path=relative_path,
                    error=str(e),
                )

        return documents

    async def sync_repository(self, repo_config: RepositoryConfig, since_commit: str) -> dict:
        """Incrementally sync a repository since a given commit.

        Returns a summary dict with indexed, deleted, and skipped counts.
        """
        scanner = GitRepoScanner(repo_config)
        resolver = self._url_resolver or URLResolver(repo_config)
        current_commit = scanner.get_current_commit()
        remote_url = scanner.get_remote_url()

        changed_files = scanner.get_changed_files(since_commit)
        deleted_files = scanner.get_deleted_files(since_commit)

        logger.info(
            "Syncing repository",
            repo=repo_config.name,
            changed=len(changed_files),
            deleted=len(deleted_files),
            since=since_commit[:8],
        )

        indexed = 0
        skipped = 0

        # Delete removed files
        for relative_path in deleted_files:
            external_id = f"{repo_config.name}:{relative_path}"
            existing = await self._traceability.get_document_by_external_id(external_id)
            if existing:
                await self.delete_document(existing)

        # Reindex changed files
        for relative_path in changed_files:
            try:
                content = scanner.read_file(relative_path)
                content_hash = compute_content_hash(content)

                external_id = f"{repo_config.name}:{relative_path}"
                existing = await self._traceability.get_document_by_external_id(external_id)

                if existing and existing.content_hash == content_hash:
                    skipped += 1
                    continue

                title = Path(relative_path).stem
                frontmatter_data, body = extract_frontmatter(content)

                doc = Document(
                    id=existing.id if existing else None,
                    title=frontmatter_data.get("title", title),
                    content=content,
                    source_path=str(scanner.repo_path / relative_path),
                    external_id=external_id,
                    domain=frontmatter_data.get("domain"),
                    tags=frontmatter_data.get("tags", []),
                    metadata=frontmatter_data,
                    repo_name=repo_config.name,
                    relative_path=relative_path,
                    git_commit=current_commit,
                    git_remote_url=remote_url,
                )

                if existing:
                    await self.reindex_document(doc)
                else:
                    await self.index_document(doc)
                indexed += 1
            except Exception as e:
                logger.error(
                    "Failed to sync file",
                    path=relative_path,
                    error=str(e),
                )

        return {
            "commit": current_commit,
            "indexed": indexed,
            "deleted": len(deleted_files),
            "skipped": skipped,
        }
