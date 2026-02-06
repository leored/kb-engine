"""Core domain models and interfaces for KB-Engine."""

from kb_engine.core.exceptions import (
    ChunkingError,
    ConfigurationError,
    ExtractionError,
    KBPodError,
    RepositoryError,
    ValidationError,
)
from kb_engine.core.models import (
    Chunk,
    Document,
    DocumentReference,
    Edge,
    EdgeType,
    Embedding,
    Node,
    NodeType,
    RetrievalMode,
    RetrievalResponse,
    RepositoryConfig,
    SearchFilters,
)

__all__ = [
    # Models
    "Document",
    "Chunk",
    "Embedding",
    "Node",
    "Edge",
    "NodeType",
    "EdgeType",
    "SearchFilters",
    "DocumentReference",
    "RetrievalResponse",
    "RetrievalMode",
    "RepositoryConfig",
    # Exceptions
    "KBPodError",
    "ConfigurationError",
    "ValidationError",
    "RepositoryError",
    "ChunkingError",
    "ExtractionError",
]
