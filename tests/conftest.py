"""Pytest configuration and fixtures."""

import pytest
from uuid import uuid4

from kb_engine.core.models.document import Chunk, ChunkType, Document, DocumentStatus
from kb_engine.core.models.graph import Edge, EdgeType, Node, NodeType


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        title="Test Document",
        content="""# Test Document

## Entity: User

A user in the system.

### Attributes

- **id**: Unique identifier
- **name**: User's full name
- **email**: Email address

## Use Case: Login

### Actors
- User

### Main Flow
1. User enters credentials
2. System validates
3. User is logged in
""",
        source_path="/test/document.md",
        domain="test",
        tags=["test", "sample"],
        status=DocumentStatus.PENDING,
        repo_name="test-repo",
        relative_path="test/document.md",
    )


@pytest.fixture
def sample_chunk(sample_document: Document) -> Chunk:
    """Create a sample chunk for testing."""
    return Chunk(
        id=uuid4(),
        document_id=sample_document.id,
        content="A user in the system with attributes like id, name, and email.",
        chunk_type=ChunkType.ENTITY,
        sequence=0,
        heading_path=["Test Document", "Entity: User"],
        section_anchor="entity-user",
    )


@pytest.fixture
def sample_node() -> Node:
    """Create a sample node for testing."""
    return Node(
        id=uuid4(),
        name="User",
        node_type=NodeType.ENTITY,
        description="A user in the system",
        properties={"domain": "test"},
    )


@pytest.fixture
def sample_edge(sample_node: Node) -> Edge:
    """Create a sample edge for testing."""
    target_node = Node(
        id=uuid4(),
        name="Login",
        node_type=NodeType.USE_CASE,
    )
    return Edge(
        id=uuid4(),
        source_id=sample_node.id,
        target_id=target_node.id,
        edge_type=EdgeType.PERFORMS,
    )
