"""Tests for SQLite traceability repository."""

import pytest
from uuid import uuid4

from kb_engine.core.models.document import Chunk, ChunkType, Document, DocumentStatus
from kb_engine.repositories.traceability.sqlite import SQLiteRepository


@pytest.fixture
async def sqlite_repo(tmp_path) -> SQLiteRepository:
    """Create a SQLite repository for testing."""
    db_path = str(tmp_path / "test.db")
    repo = SQLiteRepository(db_path=db_path)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.mark.unit
class TestSQLiteRepository:
    """Tests for SQLiteRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get_document(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(
            title="Test Doc",
            content="# Test\n\nContent here.",
            domain="test",
            tags=["entity"],
        )
        saved = await sqlite_repo.save_document(doc)
        assert saved.id == doc.id

        fetched = await sqlite_repo.get_document(doc.id)
        assert fetched is not None
        assert fetched.title == "Test Doc"
        assert fetched.domain == "test"
        assert fetched.tags == ["entity"]

    @pytest.mark.asyncio
    async def test_get_document_by_external_id(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(
            title="Test",
            content="Content",
            external_id="repo:path/to/file.md",
        )
        await sqlite_repo.save_document(doc)

        fetched = await sqlite_repo.get_document_by_external_id("repo:path/to/file.md")
        assert fetched is not None
        assert fetched.id == doc.id

    @pytest.mark.asyncio
    async def test_get_document_by_relative_path(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(
            title="Test",
            content="Content",
            repo_name="my-repo",
            relative_path="docs/entity.md",
        )
        await sqlite_repo.save_document(doc)

        fetched = await sqlite_repo.get_document_by_relative_path("my-repo", "docs/entity.md")
        assert fetched is not None
        assert fetched.id == doc.id

    @pytest.mark.asyncio
    async def test_list_documents(self, sqlite_repo: SQLiteRepository) -> None:
        for i in range(3):
            await sqlite_repo.save_document(
                Document(title=f"Doc {i}", content=f"Content {i}")
            )

        docs = await sqlite_repo.list_documents()
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_update_document(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(title="Original", content="Content")
        await sqlite_repo.save_document(doc)

        doc.title = "Updated"
        doc.status = DocumentStatus.INDEXED
        await sqlite_repo.update_document(doc)

        fetched = await sqlite_repo.get_document(doc.id)
        assert fetched is not None
        assert fetched.title == "Updated"
        assert fetched.status == DocumentStatus.INDEXED

    @pytest.mark.asyncio
    async def test_delete_document(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(title="To Delete", content="Content")
        await sqlite_repo.save_document(doc)

        deleted = await sqlite_repo.delete_document(doc.id)
        assert deleted is True

        fetched = await sqlite_repo.get_document(doc.id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_save_and_get_chunks(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(title="Doc", content="Content")
        await sqlite_repo.save_document(doc)

        chunks = [
            Chunk(
                document_id=doc.id,
                content="Chunk 1",
                chunk_type=ChunkType.ENTITY,
                sequence=0,
                heading_path=["Doc", "Section 1"],
                section_anchor="section-1",
            ),
            Chunk(
                document_id=doc.id,
                content="Chunk 2",
                chunk_type=ChunkType.DEFAULT,
                sequence=1,
                heading_path=["Doc", "Section 2"],
                section_anchor="section-2",
            ),
        ]

        saved = await sqlite_repo.save_chunks(chunks)
        assert len(saved) == 2

        fetched = await sqlite_repo.get_chunks_by_document(doc.id)
        assert len(fetched) == 2
        assert fetched[0].content == "Chunk 1"
        assert fetched[0].section_anchor == "section-1"
        assert fetched[1].content == "Chunk 2"

    @pytest.mark.asyncio
    async def test_get_chunk(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(title="Doc", content="Content")
        await sqlite_repo.save_document(doc)

        chunk = Chunk(document_id=doc.id, content="Test chunk")
        await sqlite_repo.save_chunks([chunk])

        fetched = await sqlite_repo.get_chunk(chunk.id)
        assert fetched is not None
        assert fetched.content == "Test chunk"

    @pytest.mark.asyncio
    async def test_delete_chunks_by_document(self, sqlite_repo: SQLiteRepository) -> None:
        doc = Document(title="Doc", content="Content")
        await sqlite_repo.save_document(doc)

        chunks = [
            Chunk(document_id=doc.id, content="Chunk 1", sequence=0),
            Chunk(document_id=doc.id, content="Chunk 2", sequence=1),
        ]
        await sqlite_repo.save_chunks(chunks)

        deleted = await sqlite_repo.delete_chunks_by_document(doc.id)
        assert deleted == 2

        remaining = await sqlite_repo.get_chunks_by_document(doc.id)
        assert len(remaining) == 0
