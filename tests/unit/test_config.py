from pathlib import Path

from app.core.config import Settings


class TestSettings:
    def test_defaults_match_legacy_runtime(self):
        settings = Settings(_env_file=None)

        assert settings.face_recognition_model == "VGG-Face"
        assert settings.face_detector_backend == "retinaface"
        assert settings.upload_dir == Path("images")

    def test_values_are_read_from_prefixed_environment(self, monkeypatch):
        monkeypatch.setenv("APP_FACE_DETECTOR_BACKEND", "opencv")
        monkeypatch.setenv("APP_UPLOAD_DIR", "/tmp/uploads")

        settings = Settings(_env_file=None)

        assert settings.face_detector_backend == "opencv"
        assert settings.upload_dir == Path("/tmp/uploads")
