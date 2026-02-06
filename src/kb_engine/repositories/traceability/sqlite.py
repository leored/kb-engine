"""SQLite implementation of the traceability repository."""

import json
from pathlib import Path
from uuid import UUID

import aiosqlite
import structlog

from kb_engine.core.models.document import Chunk, ChunkType, Document, DocumentStatus
from kb_engine.core.models.search import SearchFilters

logger = structlog.get_logger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    external_id TEXT UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_path TEXT,
    mime_type TEXT DEFAULT 'text/markdown',
    metadata TEXT DEFAULT '{}',
    tags TEXT DEFAULT '[]',
    domain TEXT,
    repo_name TEXT,
    relative_path TEXT,
    git_commit TEXT,
    git_remote_url TEXT,
    status TEXT DEFAULT 'pending',
    content_hash TEXT,
    created_at TEXT,
    updated_at TEXT,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    sequence INTEGER DEFAULT 0,
    content TEXT NOT NULL,
    chunk_type TEXT DEFAULT 'default',
    start_offset INTEGER,
    end_offset INTEGER,
    heading_path TEXT DEFAULT '[]',
    section_anchor TEXT,
    metadata TEXT DEFAULT '{}',
    token_count INTEGER,
    embedding_id TEXT,
    created_at TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_external_id ON documents(external_id);
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_repo_name ON documents(repo_name);
CREATE INDEX IF NOT EXISTS idx_documents_relative_path ON documents(relative_path);
"""


class SQLiteRepository:
    """SQLite implementation for document and chunk storage.

    Uses aiosqlite for async SQLite operations. Stores both
    traceability data and graph data in the same database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(CREATE_TABLES_SQL)
        await self._db.commit()
        logger.info("SQLite traceability repository initialized", db_path=self._db_path)

    async def _ensure_connected(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        return self._db

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # --- Document operations ---

    async def save_document(self, document: Document) -> Document:
        db = await self._ensure_connected()
        await db.execute(
            """INSERT OR REPLACE INTO documents
            (id, external_id, title, content, source_path, mime_type,
             metadata, tags, domain, repo_name, relative_path,
             git_commit, git_remote_url, status, content_hash,
             created_at, updated_at, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(document.id),
                document.external_id,
                document.title,
                document.content,
                document.source_path,
                document.mime_type,
                json.dumps(document.metadata),
                json.dumps(document.tags),
                document.domain,
                document.repo_name,
                document.relative_path,
                document.git_commit,
                document.git_remote_url,
                document.status.value,
                document.content_hash,
                document.created_at.isoformat(),
                document.updated_at.isoformat(),
                document.indexed_at.isoformat() if document.indexed_at else None,
            ),
        )
        await db.commit()
        return document

    async def get_document(self, document_id: UUID) -> Document | None:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "SELECT * FROM documents WHERE id = ?", (str(document_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    async def get_document_by_external_id(self, external_id: str) -> Document | None:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "SELECT * FROM documents WHERE external_id = ?", (external_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    async def get_document_by_relative_path(
        self, repo_name: str, relative_path: str
    ) -> Document | None:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "SELECT * FROM documents WHERE repo_name = ? AND relative_path = ?",
            (repo_name, relative_path),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    async def list_documents(
        self,
        filters: SearchFilters | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        db = await self._ensure_connected()
        query = "SELECT * FROM documents"
        params: list = []
        conditions: list[str] = []

        if filters:
            if filters.domains:
                placeholders = ",".join("?" * len(filters.domains))
                conditions.append(f"domain IN ({placeholders})")
                params.extend(filters.domains)
            if filters.document_ids:
                placeholders = ",".join("?" * len(filters.document_ids))
                conditions.append(f"id IN ({placeholders})")
                params.extend(str(d) for d in filters.document_ids)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_document(row) for row in rows]

    async def update_document(self, document: Document) -> Document:
        db = await self._ensure_connected()
        await db.execute(
            """UPDATE documents SET
                title=?, content=?, source_path=?, metadata=?, tags=?,
                domain=?, repo_name=?, relative_path=?, git_commit=?,
                git_remote_url=?, status=?, content_hash=?,
                updated_at=?, indexed_at=?
            WHERE id=?""",
            (
                document.title,
                document.content,
                document.source_path,
                json.dumps(document.metadata),
                json.dumps(document.tags),
                document.domain,
                document.repo_name,
                document.relative_path,
                document.git_commit,
                document.git_remote_url,
                document.status.value,
                document.content_hash,
                document.updated_at.isoformat(),
                document.indexed_at.isoformat() if document.indexed_at else None,
                str(document.id),
            ),
        )
        await db.commit()
        return document

    async def delete_document(self, document_id: UUID) -> bool:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "DELETE FROM documents WHERE id = ?", (str(document_id),)
        )
        await db.commit()
        return cursor.rowcount > 0

    # --- Chunk operations ---

    async def save_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        db = await self._ensure_connected()
        for chunk in chunks:
            await db.execute(
                """INSERT OR REPLACE INTO chunks
                (id, document_id, sequence, content, chunk_type,
                 start_offset, end_offset, heading_path, section_anchor,
                 metadata, token_count, embedding_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(chunk.id),
                    str(chunk.document_id),
                    chunk.sequence,
                    chunk.content,
                    chunk.chunk_type.value,
                    chunk.start_offset,
                    chunk.end_offset,
                    json.dumps(chunk.heading_path),
                    chunk.section_anchor,
                    json.dumps(chunk.metadata),
                    chunk.token_count,
                    str(chunk.embedding_id) if chunk.embedding_id else None,
                    chunk.created_at.isoformat(),
                ),
            )
        await db.commit()
        return chunks

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY sequence",
            (str(document_id),),
        )
        rows = await cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "SELECT * FROM chunks WHERE id = ?", (str(chunk_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    async def delete_chunks_by_document(self, document_id: UUID) -> int:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "DELETE FROM chunks WHERE document_id = ?", (str(document_id),)
        )
        await db.commit()
        return cursor.rowcount

    # --- Row mapping ---

    @staticmethod
    def _row_to_document(row: aiosqlite.Row) -> Document:
        from datetime import datetime

        return Document(
            id=UUID(row["id"]),
            external_id=row["external_id"],
            title=row["title"],
            content=row["content"],
            source_path=row["source_path"],
            mime_type=row["mime_type"] or "text/markdown",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            tags=json.loads(row["tags"]) if row["tags"] else [],
            domain=row["domain"],
            repo_name=row["repo_name"],
            relative_path=row["relative_path"],
            git_commit=row["git_commit"],
            git_remote_url=row["git_remote_url"],
            status=DocumentStatus(row["status"]),
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
        )

    @staticmethod
    def _row_to_chunk(row: aiosqlite.Row) -> Chunk:
        from datetime import datetime

        return Chunk(
            id=UUID(row["id"]),
            document_id=UUID(row["document_id"]),
            sequence=row["sequence"],
            content=row["content"],
            chunk_type=ChunkType(row["chunk_type"]),
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            heading_path=json.loads(row["heading_path"]) if row["heading_path"] else [],
            section_anchor=row["section_anchor"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            token_count=row["token_count"],
            embedding_id=UUID(row["embedding_id"]) if row["embedding_id"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )
