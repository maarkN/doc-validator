import pytest
from pydantic import ValidationError

from app.schemas.face_verification import (
    FaceVerificationRequest,
    FaceVerificationResult,
)


class TestFaceVerificationRequest:
    def test_only_images_are_required(self):
        request = FaceVerificationRequest(face_img="a", doc_img="b")

        assert request.detect_fraud is False
        assert request.detect_face_attributes is None
        assert request.remove_image is False

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError, match="surprise"):
            FaceVerificationRequest(face_img="a", doc_img="b", surprise=1)

    def test_invalid_attribute_is_rejected(self):
        with pytest.raises(ValidationError, match="detect_face_attributes"):
            FaceVerificationRequest(
                face_img="a", doc_img="b", detect_face_attributes=["height"]
            )


class TestFaceVerificationResult:
    def test_optional_fields_default_to_none_like_legacy_payload(self):
        result = FaceVerificationResult(
            verified=True, similarity_distance=0.42, similarity_threshold=0.68
        )

        assert result.model_dump() == {
            "verified": True,
            "similarity_distance": 0.42,
            "similarity_threshold": 0.68,
            "fake_face": None,
            "fake_score": None,
            "face_attributes": None,
        }
