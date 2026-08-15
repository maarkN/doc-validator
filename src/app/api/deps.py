"""Shared dependencies for the API layer."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.deepface_engine import DeepFaceEngine
from app.services.face_verification import FaceVerificationService

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def _get_engine(recognition_model: str, detector_backend: str) -> DeepFaceEngine:
    return DeepFaceEngine(recognition_model, detector_backend)


def get_face_verification_service(settings: SettingsDep) -> FaceVerificationService:
    """Provide the face verification service wired to the configured engine."""
    engine = _get_engine(
        settings.face_recognition_model, settings.face_detector_backend
    )
    return FaceVerificationService(engine, deletable_dir=settings.upload_dir)


FaceVerificationServiceDep = Annotated[
    FaceVerificationService, Depends(get_face_verification_service)
]
