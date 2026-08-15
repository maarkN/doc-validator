import base64

import cv2
import numpy as np
import pytest
import requests

from app.core.exceptions import ImageDownloadError, InvalidImageError
from app.services import image_loading
from app.services.image_loading import is_local_file, load_image


@pytest.fixture()
def jpeg_bytes() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((8, 8, 3), 200, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


class TestLoadImage:
    def test_loads_from_file_path(self, tmp_path, jpeg_bytes):
        path = tmp_path / "img.jpg"
        path.write_bytes(jpeg_bytes)

        image = load_image(str(path))

        assert image.shape == (8, 8, 3)

    def test_loads_from_bare_base64(self, jpeg_bytes):
        image = load_image(base64.b64encode(jpeg_bytes).decode())

        assert image.shape == (8, 8, 3)

    def test_loads_from_data_uri_base64(self, jpeg_bytes):
        encoded = base64.b64encode(jpeg_bytes).decode()

        image = load_image(f"data:image/jpeg;base64,{encoded}")

        assert image.shape == (8, 8, 3)

    def test_loads_from_url(self, monkeypatch, jpeg_bytes):
        class FakeResponse:
            content = jpeg_bytes

            def raise_for_status(self):
                pass

        requested = {}

        def fake_get(url, timeout):
            requested["url"] = url
            return FakeResponse()

        monkeypatch.setattr(image_loading.requests, "get", fake_get)

        image = load_image("https://example.com/img.jpg")

        assert requested["url"] == "https://example.com/img.jpg"
        assert image.shape == (8, 8, 3)

    def test_failed_download_raises_download_error(self, monkeypatch):
        def fake_get(url, timeout):
            raise requests.ConnectionError("boom")

        monkeypatch.setattr(image_loading.requests, "get", fake_get)

        with pytest.raises(ImageDownloadError):
            load_image("https://example.com/img.jpg")

    def test_undecodable_bytes_raise_invalid_image(self):
        garbage = base64.b64encode(b"not an image").decode()

        with pytest.raises(InvalidImageError):
            load_image(garbage)

    def test_invalid_base64_raises_invalid_image(self):
        with pytest.raises(InvalidImageError):
            load_image("definitely not base64!!")


class TestIsLocalFile:
    def test_existing_file_is_local(self, tmp_path):
        path = tmp_path / "img.jpg"
        path.write_bytes(b"x")

        assert is_local_file(str(path)) is True

    def test_url_and_base64_are_not_local(self):
        assert is_local_file("https://example.com/img.jpg") is False
        assert is_local_file("aGVsbG8=") is False
