"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kb_engine import __version__
from kb_engine.api.routers import admin, curation, health, indexing, retrieval
from kb_engine.config import get_settings
from kb_engine.config.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    configure_logging(
        log_level=settings.log_level,
        json_logs=settings.is_production,
    )

    # Services are lazily initialized on first request via dependencies

    yield

    # Cleanup
    if hasattr(app.state, "repo_factory"):
        await app.state.repo_factory.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="KB-Engine",
        description="Intelligent document retrieval system",
        version=__version__,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(retrieval.router, prefix="/api/v1", tags=["Retrieval"])
    app.include_router(indexing.router, prefix="/api/v1", tags=["Indexing"])
    app.include_router(curation.router, prefix="/api/v1", tags=["Curation"])
    app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])

    return app


# Create default app instance
app = create_app()


def run() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "kb_engine.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.is_development,
    )


if __name__ == "__main__":
    run()
