"""Tests for core domain models."""

import pytest
from uuid import uuid4

from kb_engine.core.models.document import Chunk, ChunkType, Document, DocumentStatus
from kb_engine.core.models.embedding import Embedding
from kb_engine.core.models.graph import Edge, EdgeType, Node, NodeType
from kb_engine.core.models.search import (
    DocumentReference,
    RetrievalMode,
    RetrievalResponse,
    SearchFilters,
)
from kb_engine.core.models.repository import RepositoryConfig


@pytest.mark.unit
class TestDocument:
    """Tests for Document model."""

    def test_create_document(self) -> None:
        doc = Document(title="Test", content="Test content")
        assert doc.title == "Test"
        assert doc.content == "Test content"
        assert doc.status == DocumentStatus.PENDING
        assert doc.id is not None

    def test_document_with_metadata(self) -> None:
        doc = Document(
            title="Test",
            content="Content",
            domain="test-domain",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        assert doc.domain == "test-domain"
        assert doc.tags == ["tag1", "tag2"]
        assert doc.metadata == {"key": "value"}

    def test_document_git_fields(self) -> None:
        doc = Document(
            title="Test",
            content="Content",
            repo_name="my-repo",
            relative_path="docs/test.md",
            git_commit="abc123",
            git_remote_url="https://github.com/org/repo",
        )
        assert doc.repo_name == "my-repo"
        assert doc.relative_path == "docs/test.md"
        assert doc.git_commit == "abc123"
        assert doc.git_remote_url == "https://github.com/org/repo"


@pytest.mark.unit
class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self) -> None:
        doc_id = uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="Chunk content",
            chunk_type=ChunkType.ENTITY,
        )
        assert chunk.document_id == doc_id
        assert chunk.content == "Chunk content"
        assert chunk.chunk_type == ChunkType.ENTITY

    def test_chunk_with_heading_path(self) -> None:
        chunk = Chunk(
            document_id=uuid4(),
            content="Content",
            heading_path=["Section 1", "Subsection 1.1"],
        )
        assert chunk.heading_path == ["Section 1", "Subsection 1.1"]

    def test_chunk_section_anchor(self) -> None:
        chunk = Chunk(
            document_id=uuid4(),
            content="Content",
            section_anchor="atributos",
        )
        assert chunk.section_anchor == "atributos"


@pytest.mark.unit
class TestDocumentReference:
    """Tests for DocumentReference model."""

    def test_create_reference(self) -> None:
        ref = DocumentReference(
            url="file:///path/to/doc.md#atributos",
            document_path="docs/entities/Usuario.md",
            section_anchor="atributos",
            title="Usuario",
            section_title="Atributos",
            score=0.92,
            snippet="Representa a una persona que interactúa con el sistema.",
            domain="core",
            tags=["entity"],
            chunk_type="entity",
            retrieval_mode=RetrievalMode.VECTOR,
        )
        assert ref.url == "file:///path/to/doc.md#atributos"
        assert ref.section_anchor == "atributos"
        assert ref.score == 0.92

    def test_reference_defaults(self) -> None:
        ref = DocumentReference(
            url="file:///doc.md",
            document_path="doc.md",
            title="Doc",
        )
        assert ref.section_anchor is None
        assert ref.score == 0.0
        assert ref.snippet == ""
        assert ref.tags == []
        assert ref.retrieval_mode == RetrievalMode.VECTOR


@pytest.mark.unit
class TestRetrievalResponse:
    """Tests for RetrievalResponse model."""

    def test_create_response(self) -> None:
        ref = DocumentReference(
            url="file:///doc.md",
            document_path="doc.md",
            title="Doc",
            score=0.8,
        )
        response = RetrievalResponse(
            query="test query",
            references=[ref],
            total_count=1,
            processing_time_ms=42.5,
        )
        assert response.query == "test query"
        assert len(response.references) == 1
        assert response.total_count == 1
        assert response.processing_time_ms == 42.5


@pytest.mark.unit
class TestRepositoryConfig:
    """Tests for RepositoryConfig model."""

    def test_create_config(self) -> None:
        config = RepositoryConfig(
            name="my-repo",
            local_path="/path/to/repo",
        )
        assert config.name == "my-repo"
        assert config.branch == "main"
        assert config.include_patterns == ["**/*.md"]
        assert config.exclude_patterns == []

    def test_config_with_remote(self) -> None:
        config = RepositoryConfig(
            name="my-repo",
            local_path="/path/to/repo",
            remote_url="https://github.com/org/repo",
            base_url_template="{remote}/blob/{branch}/{path}",
        )
        assert config.remote_url == "https://github.com/org/repo"


@pytest.mark.unit
class TestEmbedding:
    """Tests for Embedding model."""

    def test_create_embedding(self) -> None:
        chunk_id = uuid4()
        doc_id = uuid4()
        vector = [0.1] * 384

        embedding = Embedding(
            chunk_id=chunk_id,
            document_id=doc_id,
            vector=vector,
            model="all-MiniLM-L6-v2",
            dimensions=384,
        )
        assert embedding.chunk_id == chunk_id
        assert len(embedding.vector) == 384

    def test_embedding_payload(self) -> None:
        chunk_id = uuid4()
        doc_id = uuid4()

        embedding = Embedding(
            chunk_id=chunk_id,
            document_id=doc_id,
            vector=[0.1],
            model="test-model",
            dimensions=1,
            metadata={"key": "value"},
        )

        payload = embedding.payload
        assert payload["chunk_id"] == str(chunk_id)
        assert payload["document_id"] == str(doc_id)
        assert payload["model"] == "test-model"
        assert payload["key"] == "value"


@pytest.mark.unit
class TestNode:
    """Tests for Node model."""

    def test_create_node(self) -> None:
        node = Node(name="TestEntity", node_type=NodeType.ENTITY, description="A test entity")
        assert node.name == "TestEntity"
        assert node.node_type == NodeType.ENTITY

    def test_node_types(self) -> None:
        for node_type in NodeType:
            node = Node(name="Test", node_type=node_type)
            assert node.node_type == node_type


@pytest.mark.unit
class TestEdge:
    """Tests for Edge model."""

    def test_create_edge(self) -> None:
        source_id = uuid4()
        target_id = uuid4()
        edge = Edge(source_id=source_id, target_id=target_id, edge_type=EdgeType.DEPENDS_ON)
        assert edge.source_id == source_id
        assert edge.edge_type == EdgeType.DEPENDS_ON

    def test_edge_types(self) -> None:
        for edge_type in EdgeType:
            edge = Edge(source_id=uuid4(), target_id=uuid4(), edge_type=edge_type)
            assert edge.edge_type == edge_type


@pytest.mark.unit
class TestSearchFilters:
    """Tests for SearchFilters model."""

    def test_empty_filters(self) -> None:
        filters = SearchFilters()
        assert filters.document_ids is None
        assert filters.domains is None

    def test_filters_with_values(self) -> None:
        doc_id = uuid4()
        filters = SearchFilters(
            document_ids=[doc_id],
            domains=["domain1"],
            tags=["tag1"],
        )
        assert filters.document_ids == [doc_id]
        assert filters.domains == ["domain1"]
