"""Face verification endpoints.

Endpoints are plain ``def`` on purpose: the underlying inference and image
loading are blocking, so FastAPI runs them in its threadpool.
"""

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, UploadFile

from app.api.deps import FaceVerificationServiceDep, SettingsDep
from app.schemas.errors import ErrorResponse
from app.schemas.face_verification import (
    FaceVerificationRequest,
    FaceVerificationResult,
)

router = APIRouter(tags=["face-verification"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Image could not be processed"},
    422: {"model": ErrorResponse, "description": "Invalid request payload"},
    500: {"model": ErrorResponse, "description": "Unexpected internal error"},
}


@router.post(
    "/verify-faces",
    response_model=FaceVerificationResult,
    status_code=200,
    responses=_ERROR_RESPONSES,
    summary="Verify a selfie against a document photo",
)
def verify_faces(
    payload: FaceVerificationRequest,
    service: FaceVerificationServiceDep,
) -> FaceVerificationResult:
    """Compare the largest face found in each image."""
    return service.verify(payload)


@router.post(
    "/verify-faces/image",
    response_model=FaceVerificationResult,
    status_code=200,
    responses=_ERROR_RESPONSES,
    summary="Verify uploaded selfie and document images",
)
def verify_faces_with_upload(
    face: UploadFile,
    document: UploadFile,
    service: FaceVerificationServiceDep,
    settings: SettingsDep,
) -> FaceVerificationResult:
    """Store the uploads temporarily and compare them like the JSON endpoint."""
    face_path: Path | None = None
    document_path: Path | None = None
    try:
        face_path = _save_upload(face, settings.upload_dir)
        document_path = _save_upload(document, settings.upload_dir)
        return service.verify(
            FaceVerificationRequest(face_img=str(face_path), doc_img=str(document_path))
        )
    finally:
        for saved_path in (face_path, document_path):
            if saved_path is not None:
                saved_path.unlink(missing_ok=True)


def _save_upload(upload: UploadFile, upload_dir: Path) -> Path:
    destination = upload_dir / f"{uuid4()}.jpg"
    try:
        with destination.open("wb") as target:
            shutil.copyfileobj(upload.file, target)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination
