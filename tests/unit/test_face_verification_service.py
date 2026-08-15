import base64

import cv2
import numpy as np
import pytest

from app.core.exceptions import InvalidImageError
from app.schemas.face_verification import FaceVerificationRequest
from app.services.face_verification import FaceVerificationService


class FakeEngine:
    def __init__(self):
        self.extract_faces_calls = []
        self.verify_calls = []
        self.analyze_calls = []
        self.faces = [
            {
                "face": np.zeros((10, 10, 3)),
                "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
                "is_real": True,
                "antispoof_score": np.float32(0.9),
            }
        ]
        self.verify_result = {"verified": True, "distance": 0.42, "threshold": 0.68}
        self.analyze_result = [
            {"age": np.int64(30), "emotion": {"happy": np.float32(99.5)}}
        ]

    def extract_faces(self, image, anti_spoofing):
        self.extract_faces_calls.append(
            {"image": image, "anti_spoofing": anti_spoofing}
        )
        return list(self.faces)

    def verify(self, face_crop, doc_crop):
        self.verify_calls.append({"face": face_crop, "doc": doc_crop})
        return dict(self.verify_result)

    def analyze(self, image, actions):
        self.analyze_calls.append({"image": image, "actions": actions})
        return list(self.analyze_result)


@pytest.fixture()
def engine():
    return FakeEngine()


@pytest.fixture()
def service(engine, tmp_path):
    return FaceVerificationService(engine, deletable_dir=tmp_path)


@pytest.fixture()
def image_base64() -> str:
    ok, buffer = cv2.imencode(".jpg", np.full((8, 8, 3), 200, dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode()


def make_request(image_base64: str, **overrides) -> FaceVerificationRequest:
    return FaceVerificationRequest(
        face_img=image_base64, doc_img=image_base64, **overrides
    )


class TestVerify:
    def test_returns_verification_outcome(self, service, image_base64):
        result = service.verify(make_request(image_base64))

        assert result.model_dump() == {
            "verified": True,
            "similarity_distance": 0.42,
            "similarity_threshold": 0.68,
            "fake_face": None,
            "fake_score": None,
            "face_attributes": None,
        }

    def test_largest_face_of_each_image_is_compared(
        self, service, engine, image_base64
    ):
        small = {"face": "small", "facial_area": {"w": 5, "h": 5}}
        large = {"face": "large", "facial_area": {"w": 20, "h": 20}}
        engine.faces = [small, large]

        service.verify(make_request(image_base64))

        assert engine.verify_calls[0] == {"face": "large", "doc": "large"}

    def test_detect_fraud_enables_antispoofing_only_for_selfie(
        self, service, engine, image_base64
    ):
        result = service.verify(make_request(image_base64, detect_fraud=True))

        assert engine.extract_faces_calls[0]["anti_spoofing"] is True
        assert engine.extract_faces_calls[1]["anti_spoofing"] is False
        assert result.fake_face is False
        assert result.fake_score == pytest.approx(0.1)
        assert isinstance(result.fake_score, float)

    def test_face_attributes_are_json_safe(self, service, engine, image_base64):
        result = service.verify(
            make_request(image_base64, detect_face_attributes=["age", "emotion"])
        )

        assert engine.analyze_calls[0]["actions"] == ["age", "emotion"]
        assert result.face_attributes == {"age": 30, "emotion": {"happy": 99.5}}
        assert type(result.face_attributes["age"]) is int
        assert type(result.face_attributes["emotion"]["happy"]) is float

    def test_remove_image_deletes_only_local_files(self, service, tmp_path):
        ok, buffer = cv2.imencode(".jpg", np.full((8, 8, 3), 200, dtype=np.uint8))
        assert ok
        face_path = tmp_path / "face.jpg"
        doc_path = tmp_path / "doc.jpg"
        face_path.write_bytes(buffer.tobytes())
        doc_path.write_bytes(buffer.tobytes())

        service.verify(
            FaceVerificationRequest(
                face_img=str(face_path), doc_img=str(doc_path), remove_image=True
            )
        )

        assert not face_path.exists()
        assert not doc_path.exists()

    def test_remove_image_never_deletes_files_outside_the_deletable_dir(
        self, engine, tmp_path
    ):
        ok, buffer = cv2.imencode(".jpg", np.full((8, 8, 3), 200, dtype=np.uint8))
        assert ok
        protected_dir = tmp_path / "protected"
        deletable_dir = tmp_path / "deletable"
        protected_dir.mkdir()
        deletable_dir.mkdir()
        protected_file = protected_dir / "asset.jpg"
        protected_file.write_bytes(buffer.tobytes())
        service = FaceVerificationService(engine, deletable_dir=deletable_dir)

        result = service.verify(
            FaceVerificationRequest(
                face_img=str(protected_file),
                doc_img=str(protected_file),
                remove_image=True,
            )
        )

        assert result.verified is True
        assert protected_file.exists()

    def test_remove_image_with_base64_input_is_a_no_op(self, service, image_base64):
        result = service.verify(make_request(image_base64, remove_image=True))

        assert result.verified is True

    def test_no_extracted_face_raises_invalid_image(
        self, service, engine, image_base64
    ):
        engine.faces = []

        with pytest.raises(InvalidImageError):
            service.verify(make_request(image_base64))
