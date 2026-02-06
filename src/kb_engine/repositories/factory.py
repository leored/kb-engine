"""Repository factory for creating repository instances."""

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from kb_engine.config.settings import Settings

logger = structlog.get_logger(__name__)


class RepositoryFactory:
    """Factory for creating repository instances.

    Creates the appropriate repository implementations based on
    configuration settings (profile: local vs server).
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._traceability = None
        self._vector = None
        self._graph = None

    async def get_traceability_repository(self):
        """Get or create the traceability repository."""
        if self._traceability is None:
            store = self._settings.traceability_store.lower()

            if store == "sqlite":
                from kb_engine.repositories.traceability.sqlite import SQLiteRepository

                self._traceability = SQLiteRepository(
                    db_path=self._settings.sqlite_path,
                )
                await self._traceability.initialize()
            elif store == "postgres":
                from kb_engine.repositories.traceability.postgres import PostgresRepository

                self._traceability = PostgresRepository(
                    connection_string=self._settings.database_url,
                )
            else:
                raise ValueError(f"Unknown traceability store: {store}")

            logger.info("Traceability repository created", store=store)

        return self._traceability

    async def get_vector_repository(self):
        """Get or create the vector repository."""
        if self._vector is None:
            vector_store = self._settings.vector_store.lower()

            if vector_store == "chroma":
                from kb_engine.repositories.vector.chroma import ChromaRepository

                self._vector = ChromaRepository(
                    persist_directory=self._settings.chroma_path,
                )
                await self._vector.initialize()
            elif vector_store == "qdrant":
                from kb_engine.repositories.vector.qdrant import QdrantRepository

                self._vector = QdrantRepository(
                    host=self._settings.qdrant_host,
                    port=self._settings.qdrant_port,
                    api_key=self._settings.qdrant_api_key,
                    collection_name=self._settings.qdrant_collection,
                )
            else:
                raise ValueError(f"Unknown vector store: {vector_store}")

            logger.info("Vector repository created", store=vector_store)

        return self._vector

    async def get_graph_repository(self):
        """Get or create the graph repository (optional)."""
        if self._graph is None:
            graph_store = self._settings.graph_store.lower()

            if graph_store == "none":
                return None
            elif graph_store == "sqlite":
                from kb_engine.repositories.graph.sqlite import SQLiteGraphRepository

                self._graph = SQLiteGraphRepository(
                    db_path=self._settings.sqlite_path,
                )
                await self._graph.initialize()
            elif graph_store == "neo4j":
                from kb_engine.repositories.graph.neo4j import Neo4jRepository

                self._graph = Neo4jRepository(
                    uri=self._settings.neo4j_uri,
                    user=self._settings.neo4j_user,
                    password=self._settings.neo4j_password,
                )
            else:
                raise ValueError(f"Unknown graph store: {graph_store}")

            logger.info("Graph repository created", store=graph_store)

        return self._graph

    async def close(self) -> None:
        """Close all repository connections."""
        if hasattr(self._traceability, "close"):
            await self._traceability.close()
        if hasattr(self._graph, "close"):
            await self._graph.close()
        self._traceability = None
        self._vector = None
        self._graph = None
