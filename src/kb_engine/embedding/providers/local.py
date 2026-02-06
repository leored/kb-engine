"""Local embedding provider using sentence-transformers."""

import asyncio

import structlog

from kb_engine.embedding.base import EmbeddingProvider

logger = structlog.get_logger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local model-based embedding provider.

    Uses sentence-transformers for local embedding generation.
    Runs the model in a thread pool to avoid blocking the event loop.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._model_path = model_path
        self._model_name_str = model_name
        self._model = None
        self._dimensions_cache: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name_str

    @property
    def dimensions(self) -> int:
        if self._dimensions_cache is not None:
            return self._dimensions_cache
        # Default dimensions for common models
        defaults = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
        }
        return defaults.get(self._model_name_str, 384)

    def _ensure_model(self) -> None:
        """Load the sentence-transformers model (synchronous)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_id = self._model_path or self._model_name_str
            logger.info("Loading embedding model", model=model_id)
            self._model = SentenceTransformer(model_id)
            self._dimensions_cache = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedding model loaded",
                model=model_id,
                dimensions=self._dimensions_cache,
            )

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using local model."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_text_sync, text)

    def _embed_text_sync(self, text: str) -> list[float]:
        self._ensure_model()
        assert self._model is not None
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using local model."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_texts_sync, texts)

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        assert self._model is not None
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
