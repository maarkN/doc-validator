"""Business logic for verifying a selfie against a document photo."""

import os
from pathlib import Path
from typing import Any

import numpy as np

from app.core.exceptions import InvalidImageError
from app.schemas.face_verification import (
    FaceVerificationRequest,
    FaceVerificationResult,
)
from app.services.deepface_engine import DeepFaceEngine
from app.services.image_loading import is_local_file, load_image


class FaceVerificationService:
    """Orchestrates face extraction, comparison and optional analyses."""

    def __init__(self, engine: DeepFaceEngine, deletable_dir: Path) -> None:
        """Wire the inference engine and the only directory deletions may touch.

        Args:
            engine: Performs the actual DeepFace calls.
            deletable_dir: ``remove_image`` only deletes files inside this
                directory; client-supplied paths elsewhere are never removed.
        """
        self._engine = engine
        self._deletable_dir = deletable_dir

    def verify(self, request: FaceVerificationRequest) -> FaceVerificationResult:
        """Compare the largest face of each image and build the result.

        Args:
            request: Validated verification request.

        Returns:
            The verification outcome, including anti-spoofing and facial
            attributes when requested.

        Raises:
            InvalidImageError: If an image cannot be decoded or contains no
                extractable face.
            ImageDownloadError: If an image URL cannot be fetched.
        """
        face_image = load_image(request.face_img)
        document_image = load_image(request.doc_img)

        face = self._largest_face(
            self._engine.extract_faces(face_image, anti_spoofing=request.detect_fraud)
        )
        document_face = self._largest_face(
            self._engine.extract_faces(document_image, anti_spoofing=False)
        )

        verification = self._engine.verify(face["face"], document_face["face"])
        result = FaceVerificationResult(
            verified=verification["verified"],
            similarity_distance=verification["distance"],
            similarity_threshold=verification["threshold"],
        )

        if request.detect_fraud:
            result.fake_face = not face["is_real"]
            result.fake_score = float(1 - face["antispoof_score"])
        if request.detect_face_attributes:
            attributes = self._engine.analyze(
                face_image, actions=list(request.detect_face_attributes)
            )
            result.face_attributes = _to_json_safe(attributes[0])
        if request.remove_image:
            self._remove_local_inputs(request.face_img, request.doc_img)

        return result

    @staticmethod
    def _largest_face(extracted_faces: list[dict[str, Any]]) -> dict[str, Any]:
        if not extracted_faces:
            raise InvalidImageError("No face could be extracted from the image.")

        def area(face: dict[str, Any]) -> int:
            return int(face["facial_area"]["w"]) * int(face["facial_area"]["h"])

        return max(extracted_faces, key=area)

    def _remove_local_inputs(self, *sources: str) -> None:
        for source in sources:
            if is_local_file(source) and self._is_deletable(Path(source)):
                os.remove(source)

    def _is_deletable(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._deletable_dir.resolve())
        except ValueError:
            return False
        return True


def _to_json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars so the payload survives serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    return value
