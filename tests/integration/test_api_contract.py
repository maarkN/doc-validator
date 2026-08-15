"""Contract tests for the v1 API.

Ported from the legacy characterization suite: same behavioral core
(orchestration parameters, largest-face selection, response payload shape),
with the deliberate contract changes approved for the refactoring —
/api/v1 prefix, real status codes and the {detail, code} error envelope.
"""

import io

import pytest


class TestHealth:
    def test_health_reports_online(self, client):
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"status": "online"}


class TestVerifyFacesJson:
    def test_base64_images_return_verification_payload(self, client, tiny_jpeg_base64):
        response = client.post(
            "/api/v1/verify-faces",
            json={"face_img": tiny_jpeg_base64, "doc_img": tiny_jpeg_base64},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "verified": True,
            "similarity_distance": 0.42,
            "similarity_threshold": 0.68,
            "fake_face": None,
            "fake_score": None,
            "face_attributes": None,
        }

    def test_file_path_images_are_accepted(self, client, tiny_jpeg_file):
        response = client.post(
            "/api/v1/verify-faces",
            json={"face_img": tiny_jpeg_file, "doc_img": tiny_jpeg_file},
        )

        assert response.status_code == 200
        assert response.json()["verified"] is True

    def test_remove_image_deletes_local_input_files(
        self, client, tmp_path, tiny_jpeg_bytes
    ):
        face_path = tmp_path / "selfie.jpg"
        doc_path = tmp_path / "document.jpg"
        face_path.write_bytes(tiny_jpeg_bytes)
        doc_path.write_bytes(tiny_jpeg_bytes)

        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": str(face_path),
                "doc_img": str(doc_path),
                "remove_image": True,
            },
        )

        assert response.status_code == 200
        assert not face_path.exists()
        assert not doc_path.exists()

    def test_remove_image_with_base64_input_succeeds_as_a_no_op(
        self, client, tiny_jpeg_base64
    ):
        # Legacy behavior: this crashed into a 200 api_args_error envelope.
        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "remove_image": True,
            },
        )

        assert response.status_code == 200
        assert response.json()["verified"] is True

    def test_detect_fraud_populates_fake_fields_and_enables_antispoofing(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "detect_fraud": True,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["fake_face"] is False
        assert body["fake_score"] == pytest.approx(0.1)
        assert deepface_stub.extract_faces_calls[0]["anti_spoofing"] is True
        assert deepface_stub.extract_faces_calls[1]["anti_spoofing"] is False

    def test_detect_face_attributes_returns_json_safe_attributes(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "detect_face_attributes": ["age", "emotion"],
            },
        )

        assert response.status_code == 200
        # Legacy behavior: numpy scalars crashed serialization and the whole
        # feature returned an error envelope.
        assert response.json()["face_attributes"] == {
            "age": 30,
            "emotion": {"happy": 99.5},
        }
        assert deepface_stub.analyze_calls[0]["actions"] == ["age", "emotion"]

    def test_missing_required_field_returns_422_envelope(self, client):
        response = client.post("/api/v1/verify-faces", json={"face_img": "x"})

        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "validation_error"
        assert "doc_img" in body["detail"]

    def test_unknown_extra_field_returns_422_envelope(self, client, tiny_jpeg_base64):
        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "surprise": 1,
            },
        )

        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"

    def test_invalid_attribute_value_returns_422_envelope(
        self, client, tiny_jpeg_base64
    ):
        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "detect_face_attributes": ["height"],
            },
        )

        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"

    def test_non_json_body_returns_422_envelope(self, client):
        response = client.post(
            "/api/v1/verify-faces",
            content="not json",
            headers={"content-type": "text/plain"},
        )

        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"

    def test_undecodable_base64_returns_400_invalid_image(self, client):
        response = client.post(
            "/api/v1/verify-faces",
            json={"face_img": "!!!definitely-not-an-image!!!", "doc_img": "x"},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_image"

    def test_largest_face_is_selected_for_verification(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        import numpy as np

        small_face = np.zeros((5, 5, 3))
        big_face = np.ones((20, 20, 3))
        deepface_stub.faces = [
            {"face": small_face, "facial_area": {"x": 0, "y": 0, "w": 5, "h": 5}},
            {"face": big_face, "facial_area": {"x": 0, "y": 0, "w": 20, "h": 20}},
        ]

        response = client.post(
            "/api/v1/verify-faces",
            json={"face_img": tiny_jpeg_base64, "doc_img": tiny_jpeg_base64},
        )

        assert response.status_code == 200
        verify_call = deepface_stub.verify_calls[0]
        assert verify_call["img1_path"] is big_face
        assert verify_call["img2_path"] is big_face

    def test_deepface_orchestration_parameters_are_preserved(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        client.post(
            "/api/v1/verify-faces",
            json={"face_img": tiny_jpeg_base64, "doc_img": tiny_jpeg_base64},
        )

        for call in deepface_stub.extract_faces_calls:
            assert call["detector_backend"] == "retinaface"
            assert call["enforce_detection"] is False
            assert call["align"] is True
        verify_call = deepface_stub.verify_calls[0]
        assert verify_call["model_name"] == "VGG-Face"
        assert verify_call["detector_backend"] == "skip"
        assert verify_call["enforce_detection"] is False
        assert verify_call["align"] is False


class TestVerifyFacesUpload:
    def test_multipart_upload_returns_verification_payload(
        self, client, tiny_jpeg_bytes, tmp_path
    ):
        response = client.post(
            "/api/v1/verify-faces/image",
            files={
                "document": ("doc.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg"),
                "face": ("face.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg"),
            },
        )

        assert response.status_code == 200
        assert response.json()["verified"] is True
        # Uploads are always cleaned up.
        assert list((tmp_path / "uploads").iterdir()) == []

    def test_missing_file_part_returns_422_envelope(self, client, tiny_jpeg_bytes):
        response = client.post(
            "/api/v1/verify-faces/image",
            files={"document": ("doc.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg")},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "validation_error"
        assert "face" in body["detail"]


class TestDocs:
    def test_interactive_docs_are_served(self, client):
        response = client.get("/docs")

        assert response.status_code == 200
