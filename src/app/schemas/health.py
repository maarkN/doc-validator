"""Health-check contract."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Reports that the API is up."""

    status: Literal["online"] = "online"
