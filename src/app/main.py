"""Application factory and lifespan management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router as api_v1_router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare runtime resources before serving requests."""
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with routers and handlers registered."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Verifies that a selfie and an identity-document photo "
        "show the same person.",
        lifespan=lifespan,
    )
    app.include_router(api_v1_router, prefix="/api/v1")
    return app
