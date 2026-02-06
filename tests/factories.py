"""Test factories using factory_boy."""

from uuid import uuid4

import factory

from kb_engine.core.models.document import Chunk, ChunkType, Document, DocumentStatus
from kb_engine.core.models.embedding import Embedding
from kb_engine.core.models.graph import Edge, EdgeType, Node, NodeType


class DocumentFactory(factory.Factory):
    """Factory for creating Document instances."""

    class Meta:
        model = Document

    id = factory.LazyFunction(uuid4)
    title = factory.Sequence(lambda n: f"Document {n}")
    content = factory.Faker("paragraph", nb_sentences=5)
    source_path = factory.Sequence(lambda n: f"/docs/document_{n}.md")
    domain = factory.Faker("word")
    tags = factory.LazyFunction(lambda: ["test"])
    status = DocumentStatus.PENDING


class ChunkFactory(factory.Factory):
    """Factory for creating Chunk instances."""

    class Meta:
        model = Chunk

    id = factory.LazyFunction(uuid4)
    document_id = factory.LazyFunction(uuid4)
    content = factory.Faker("paragraph", nb_sentences=2)
    chunk_type = ChunkType.DEFAULT
    sequence = factory.Sequence(lambda n: n)
    heading_path = factory.LazyFunction(lambda: ["Section"])


class EmbeddingFactory(factory.Factory):
    """Factory for creating Embedding instances."""

    class Meta:
        model = Embedding

    id = factory.LazyFunction(uuid4)
    chunk_id = factory.LazyFunction(uuid4)
    document_id = factory.LazyFunction(uuid4)
    vector = factory.LazyFunction(lambda: [0.1] * 384)
    model = "all-MiniLM-L6-v2"
    dimensions = 384


class NodeFactory(factory.Factory):
    """Factory for creating Node instances."""

    class Meta:
        model = Node

    id = factory.LazyFunction(uuid4)
    name = factory.Faker("word")
    node_type = NodeType.ENTITY
    description = factory.Faker("sentence")


class EdgeFactory(factory.Factory):
    """Factory for creating Edge instances."""

    class Meta:
        model = Edge

    id = factory.LazyFunction(uuid4)
    source_id = factory.LazyFunction(uuid4)
    target_id = factory.LazyFunction(uuid4)
    edge_type = EdgeType.RELATED_TO
