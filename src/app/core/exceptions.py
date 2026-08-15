"""Domain exceptions and their one-time translation to HTTP responses.

Services raise :class:`DomainError` subclasses; the handlers registered by
:func:`register_exception_handlers` translate them into the uniform
``{"detail": ..., "code": ...}`` envelope. No route does its own try/except.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class DomainError(Exception):
    """Base class for expected application failures.

    Attributes:
        code: Stable machine-readable identifier exposed in error responses.
        status_code: HTTP status the API layer maps this error to.
    """

    code = "internal_error"
    status_code = 500

    def __init__(self, detail: str) -> None:
        """Store the human-readable detail exposed in the error response."""
        super().__init__(detail)
        self.detail = detail


class InvalidImageError(DomainError):
    """The provided image could not be decoded or contains no usable face."""

    code = "invalid_image"
    status_code = 400


class ImageDownloadError(DomainError):
    """An image URL could not be fetched."""

    code = "image_download_failed"
    status_code = 400


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the global exception-to-HTTP translation to ``app``."""

    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, error: DomainError) -> JSONResponse:
        return _error_response(error.status_code, error.detail, error.code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        detail = "; ".join(
            f"{_field_path(item)}: {item['msg']}" for item in error.errors()
        )
        return _error_response(422, detail, "validation_error")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        return _error_response(error.status_code, str(error.detail), "http_error")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, error: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled error while processing %s", request.url.path)
        return _error_response(500, "Internal server error.", "internal_error")


def _error_response(status_code: int, detail: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"detail": detail, "code": code}
    )


def _field_path(error_item: dict[str, object]) -> str:
    location = error_item.get("loc", ())
    if isinstance(location, list | tuple):
        return ".".join(str(part) for part in location)
    return str(location)
