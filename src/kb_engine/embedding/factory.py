"""Factory for creating embedding providers."""

from kb_engine.embedding.base import EmbeddingProvider
from kb_engine.embedding.config import EmbeddingConfig


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()

    def create_provider(self) -> EmbeddingProvider:
        """Create an embedding provider based on configuration."""
        provider = self._config.provider.lower()

        if provider == "openai":
            from kb_engine.embedding.providers.openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                model=self._config.openai_model,
                dimensions=self._config.openai_dimensions,
            )
        elif provider == "local":
            from kb_engine.embedding.providers.local import LocalEmbeddingProvider

            return LocalEmbeddingProvider(
                model_path=self._config.local_model_path,
                model_name=self._config.local_model_name,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
