"""The single error envelope used by every non-2xx response."""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error body: human-readable detail plus a machine-readable code."""

    detail: str = Field(description="Human-readable description of the error.")
    code: str = Field(description="Stable machine-readable error code.")
