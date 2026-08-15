"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, sourced from the environment or a .env file.

    Every variable is prefixed with ``APP_`` (e.g. ``APP_LOG_LEVEL``).
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_", env_file=".env", extra="ignore"
    )

    app_name: str = "doc-validator"
    log_level: str = "INFO"

    face_recognition_model: str = "VGG-Face"
    face_detector_backend: str = "retinaface"

    upload_dir: Path = Path("images")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
