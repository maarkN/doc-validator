"""Freezes the observable HTTP contract of the legacy Flask app.

Every assertion here documents CURRENT behavior, including behavior that is
arguably buggy (e.g. HTTP 200 on errors). Do not "fix" an assertion without an
explicit product decision — these tests are the refactoring safety net.
"""

import json
import os


class TestIndex:
    def test_index_returns_plain_text_online_marker(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.get_data(as_text=True) == "API-ONLINE"
        assert response.mimetype == "text/html"


class TestVerifyFacesJson:
    def test_base64_images_return_verification_payload(self, client, tiny_jpeg_base64):
        response = client.post(
            "/verify-faces",
            json={"face_img": tiny_jpeg_base64, "doc_img": tiny_jpeg_base64},
        )

        assert response.status_code == 200
        # Success path returns json.dumps(...) as a Flask string -> text/html.
        assert response.mimetype == "text/html"
        body = json.loads(response.get_data(as_text=True))
        assert body == {
            "verified": True,
            "similarity_distance": 0.42,
            "similarity_threshold": 0.68,
            "fake_face": None,
            "fake_score": None,
            "face_attributes": None,
        }

    def test_file_path_images_are_accepted(self, client, tiny_jpeg_file):
        response = client.post(
            "/verify-faces",
            json={"face_img": tiny_jpeg_file, "doc_img": tiny_jpeg_file},
        )

        assert response.status_code == 200
        body = json.loads(response.get_data(as_text=True))
        assert body["verified"] is True

    def test_remove_image_true_deletes_file_path_inputs(
        self, client, tmp_path, tiny_jpeg_file
    ):
        response = client.post(
            "/verify-faces",
            json={
                "face_img": tiny_jpeg_file,
                "doc_img": tiny_jpeg_file,
                "remove_image": True,
            },
        )

        # Both args point at the same file: the second os.remove raises,
        # which the legacy app converts into the error envelope with 200.
        assert response.status_code == 200
        assert not os.path.exists(tiny_jpeg_file)
        assert "api_args_error" in response.get_json()

    def test_remove_image_true_with_two_distinct_files_succeeds(
        self, client, tmp_path, tiny_jpeg_bytes
    ):
        face_path = tmp_path / "face.jpg"
        doc_path = tmp_path / "doc.jpg"
        face_path.write_bytes(tiny_jpeg_bytes)
        doc_path.write_bytes(tiny_jpeg_bytes)

        response = client.post(
            "/verify-faces",
            json={
                "face_img": str(face_path),
                "doc_img": str(doc_path),
                "remove_image": True,
            },
        )

        assert response.status_code == 200
        body = json.loads(response.get_data(as_text=True))
        assert body["verified"] is True
        assert not face_path.exists()
        assert not doc_path.exists()

    def test_remove_image_true_with_base64_input_returns_error_envelope(
        self, client, tiny_jpeg_base64
    ):
        response = client.post(
            "/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "remove_image": True,
            },
        )

        # os.remove() is called on the base64 string itself and raises.
        assert response.status_code == 200
        assert response.mimetype == "application/json"
        assert "api_args_error" in response.get_json()

    def test_detect_fraud_populates_fake_fields_and_enables_antispoofing(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        response = client.post(
            "/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "detect_fraud": True,
            },
        )

        assert response.status_code == 200
        body = json.loads(response.get_data(as_text=True))
        assert body["fake_face"] is False
        assert body["fake_score"] == 1 - 0.9
        # anti_spoofing is enabled only for the face image, not the document.
        assert deepface_stub.extract_faces_calls[0]["anti_spoofing"] is True
        assert deepface_stub.extract_faces_calls[1]["anti_spoofing"] is False

    def test_detect_face_attributes_calls_analyze_and_returns_attributes(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        response = client.post(
            "/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "detect_face_attributes": ["age", "emotion"],
            },
        )

        assert response.status_code == 200
        body = json.loads(response.get_data(as_text=True))
        assert body["face_attributes"] == {"age": 30, "dominant_emotion": "happy"}
        assert deepface_stub.analyze_calls[0]["actions"] == ["age", "emotion"]

    def test_missing_required_field_returns_200_with_error_envelope(self, client):
        response = client.post("/verify-faces", json={"face_img": "x"})

        assert response.status_code == 200
        assert response.mimetype == "application/json"
        body = response.get_json()
        assert set(body.keys()) == {"api_args_error"}
        assert "doc_img" in body["api_args_error"]

    def test_unknown_extra_field_returns_200_with_error_envelope(
        self, client, tiny_jpeg_base64
    ):
        response = client.post(
            "/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "surprise": 1,
            },
        )

        # ApiArgsModel tolerates the extra key, but **args then hits
        # compare_face_to_doc() with an unexpected kwarg -> TypeError -> 200.
        assert response.status_code == 200
        assert "api_args_error" in response.get_json()

    def test_invalid_attribute_value_returns_200_with_error_envelope(
        self, client, tiny_jpeg_base64
    ):
        response = client.post(
            "/verify-faces",
            json={
                "face_img": tiny_jpeg_base64,
                "doc_img": tiny_jpeg_base64,
                "detect_face_attributes": ["height"],
            },
        )

        assert response.status_code == 200
        assert "api_args_error" in response.get_json()

    def test_non_json_body_returns_200_with_error_envelope(self, client):
        response = client.post(
            "/verify-faces", data="not json", content_type="text/plain"
        )

        assert response.status_code == 200
        assert "api_args_error" in response.get_json()

    def test_null_json_body_returns_200_with_error_envelope(self, client):
        response = client.post("/verify-faces", json=None)

        assert response.status_code == 200
        assert "api_args_error" in response.get_json()

    def test_largest_face_is_selected_for_verification(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        import numpy as np

        small_face = np.zeros((5, 5, 3))
        big_face = np.ones((20, 20, 3))
        deepface_stub.faces = [
            {
                "face": small_face,
                "facial_area": {"x": 0, "y": 0, "w": 5, "h": 5},
                "is_real": True,
                "antispoof_score": 0.9,
            },
            {
                "face": big_face,
                "facial_area": {"x": 0, "y": 0, "w": 20, "h": 20},
                "is_real": True,
                "antispoof_score": 0.9,
            },
        ]

        response = client.post(
            "/verify-faces",
            json={"face_img": tiny_jpeg_base64, "doc_img": tiny_jpeg_base64},
        )

        assert response.status_code == 200
        verify_call = deepface_stub.verify_calls[0]
        assert verify_call["img1_path"] is big_face
        assert verify_call["img2_path"] is big_face

    def test_deepface_orchestration_parameters(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        client.post(
            "/verify-faces",
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


class TestVerifyFacesImageUpload:
    def test_multipart_upload_returns_verification_payload(
        self, client, tiny_jpeg_bytes
    ):
        import io

        response = client.post(
            "/verify-faces/image",
            data={
                "document": (io.BytesIO(tiny_jpeg_bytes), "doc.jpg"),
                "face": (io.BytesIO(tiny_jpeg_bytes), "face.jpg"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        body = json.loads(response.get_data(as_text=True))
        assert body["verified"] is True
        # remove_image=True is hardcoded: saved uploads are cleaned up.
        assert os.listdir("images") == []

    def test_missing_file_part_returns_200_with_error_envelope(
        self, client, tiny_jpeg_bytes
    ):
        import io

        response = client.post(
            "/verify-faces/image",
            data={"document": (io.BytesIO(tiny_jpeg_bytes), "doc.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert "api_args_error" in response.get_json()


class TestDocsRoute:
    def test_swagger_docs_are_served(self, client):
        response = client.get("/docs/")

        assert response.status_code == 200
