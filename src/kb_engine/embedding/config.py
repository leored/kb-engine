"""Embedding configuration."""

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    # Provider selection
    provider: str = Field(default="local", description="Embedding provider (openai, local)")

    # OpenAI settings
    openai_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    openai_dimensions: int = Field(
        default=1536, description="Embedding dimensions for OpenAI"
    )

    # Local model settings
    local_model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence-transformers model name"
    )
    local_model_path: str | None = Field(
        default=None, description="Path to local embedding model (overrides name)"
    )

    # Batch settings
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for embedding")

    class Config:
        frozen = True
