"""Logging configuration and per-request correlation ids."""

import logging
import logging.config
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request, Response

from app.core.exceptions import error_response

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

_REQUEST_ID_HEADER = "X-Request-ID"

logger = logging.getLogger(__name__)


class CorrelationIdFilter(logging.Filter):
    """Injects the current request's correlation id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``correlation_id`` to the record; never drops records."""
        record.correlation_id = correlation_id_var.get()
        return True


def setup_logging(log_level: str) -> None:
    """Configure root logging with the correlation-id-aware formatter."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"correlation_id": {"()": CorrelationIdFilter}},
            "formatters": {
                "default": {
                    "format": (
                        "%(asctime)s %(levelname)s %(name)s "
                        "[%(correlation_id)s] %(message)s"
                    )
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["correlation_id"],
                }
            },
            "root": {"level": log_level.upper(), "handlers": ["console"]},
        }
    )


async def correlation_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a correlation id to the request and echo it back in the response.

    An incoming ``X-Request-ID`` header is honored so ids can be traced
    across services; otherwise a new id is generated.
    """
    correlation_id = request.headers.get(_REQUEST_ID_HEADER) or uuid4().hex[:12]
    token = correlation_id_var.set(correlation_id)
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Handled here, not in the global Exception handler: Starlette
            # runs that handler outside this middleware, where the
            # correlation id would already be gone from the log and header.
            logger.exception("Unhandled error while processing %s", request.url.path)
            response = error_response(500, "Internal server error.", "internal_error")
        response.headers[_REQUEST_ID_HEADER] = correlation_id
        logger.info(
            "%s %s -> %d", request.method, request.url.path, response.status_code
        )
        return response
    finally:
        correlation_id_var.reset(token)
