"""Health-check endpoint."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=200)
def health() -> HealthResponse:
    """Report that the API is up and able to serve requests."""
    return HealthResponse()
