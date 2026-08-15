import io


class TestUnexpectedErrors:
    def test_engine_failure_returns_500_envelope(
        self, client, deepface_stub, tiny_jpeg_base64
    ):
        def explode(**kwargs):
            raise RuntimeError("model blew up")

        deepface_stub.verify = explode

        response = client.post(
            "/api/v1/verify-faces",
            json={"face_img": tiny_jpeg_base64, "doc_img": tiny_jpeg_base64},
        )

        assert response.status_code == 500
        body = response.json()
        assert body == {"detail": "Internal server error.", "code": "internal_error"}
        # The concrete exception message must not leak to the client.
        assert "model blew up" not in response.text

    def test_unknown_route_returns_404_envelope(self, client):
        response = client.get("/api/v1/nope")

        assert response.status_code == 404
        assert response.json()["code"] == "http_error"


class TestUploadCleanup:
    def test_uploads_are_removed_even_when_verification_fails(
        self, client, deepface_stub, tiny_jpeg_bytes, tmp_path
    ):
        def explode(**kwargs):
            raise RuntimeError("boom")

        deepface_stub.extract_faces = explode

        response = client.post(
            "/api/v1/verify-faces/image",
            files={
                "document": ("doc.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg"),
                "face": ("face.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg"),
            },
        )

        assert response.status_code == 500
        assert list((tmp_path / "uploads").iterdir()) == []

    def test_first_upload_is_removed_when_saving_the_second_fails(
        self, client, monkeypatch, tiny_jpeg_bytes, tmp_path
    ):
        import shutil as real_shutil

        from app.api.v1.routes import face_verification as route_module

        calls = {"count": 0}

        def copy_then_fail(source, target):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("disk full")
            real_shutil.copyfileobj(source, target)

        monkeypatch.setattr(route_module.shutil, "copyfileobj", copy_then_fail)

        response = client.post(
            "/api/v1/verify-faces/image",
            files={
                "document": ("doc.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg"),
                "face": ("face.jpg", io.BytesIO(tiny_jpeg_bytes), "image/jpeg"),
            },
        )

        assert response.status_code == 500
        assert list((tmp_path / "uploads").iterdir()) == []


class TestUrlImages:
    def test_url_images_are_downloaded_and_verified(
        self, client, monkeypatch, tiny_jpeg_bytes
    ):
        from app.services import image_loading

        class FakeResponse:
            content = tiny_jpeg_bytes

            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            image_loading.requests, "get", lambda url, timeout: FakeResponse()
        )

        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": "https://example.com/selfie.jpg",
                "doc_img": "https://example.com/doc.jpg",
            },
        )

        assert response.status_code == 200
        assert response.json()["verified"] is True

    def test_unreachable_url_returns_400_download_error(self, client, monkeypatch):
        import requests

        from app.services import image_loading

        def fail(url, timeout):
            raise requests.ConnectionError("unreachable")

        monkeypatch.setattr(image_loading.requests, "get", fail)

        response = client.post(
            "/api/v1/verify-faces",
            json={
                "face_img": "https://example.com/selfie.jpg",
                "doc_img": "https://example.com/doc.jpg",
            },
        )

        assert response.status_code == 400
        assert response.json()["code"] == "image_download_failed"


class TestCorrelationId:
    def test_response_carries_a_generated_request_id(self, client):
        response = client.get("/api/v1/health")

        assert response.headers.get("X-Request-ID")

    def test_incoming_request_id_is_echoed_back(self, client):
        response = client.get(
            "/api/v1/health", headers={"X-Request-ID": "trace-abc-123"}
        )

        assert response.headers["X-Request-ID"] == "trace-abc-123"
