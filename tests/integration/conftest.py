"""Integration-test fixtures: real app, DeepFace stubbed at the boundary.

The DeepFace module is replaced in ``sys.modules`` before the engine's lazy
imports run, so these tests exercise the full HTTP stack (routing,
validation, service orchestration, error translation, serialization)
without TensorFlow or model weights.
"""

import base64
import sys
import types

import cv2
import numpy as np
import pytest

from app.api.deps import _get_engine


class DeepFaceStub:
    """Records calls and returns canned, controllable responses."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.extract_faces_calls = []
        self.verify_calls = []
        self.analyze_calls = []
        self.faces = [
            {
                "face": np.zeros((10, 10, 3), dtype=np.float64),
                "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
                "confidence": 0.99,
                "is_real": True,
                "antispoof_score": np.float32(0.9),
            }
        ]
        self.verify_result = {
            "verified": True,
            "distance": np.float64(0.42),
            "threshold": 0.68,
        }
        self.analyze_result = [
            {"age": np.int64(30), "emotion": {"happy": np.float32(99.5)}}
        ]

    def extract_faces(self, **kwargs):
        self.extract_faces_calls.append(kwargs)
        return list(self.faces)

    def verify(self, **kwargs):
        self.verify_calls.append(kwargs)
        return dict(self.verify_result)

    def analyze(self, **kwargs):
        self.analyze_calls.append(kwargs)
        return list(self.analyze_result)


_stub = DeepFaceStub()

_deepface_module = types.ModuleType("deepface")
_deepface_module.DeepFace = _stub
sys.modules.setdefault("deepface", _deepface_module)


@pytest.fixture()
def deepface_stub():
    _stub.reset()
    return _stub


@pytest.fixture()
def client(deepface_stub, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("APP_UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    _get_engine.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture()
def tiny_jpeg_bytes() -> bytes:
    image = np.full((12, 12, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def tiny_jpeg_base64(tiny_jpeg_bytes) -> str:
    return base64.b64encode(tiny_jpeg_bytes).decode("utf-8")


@pytest.fixture()
def tiny_jpeg_file(tmp_path, tiny_jpeg_bytes) -> str:
    path = tmp_path / "face.jpg"
    path.write_bytes(tiny_jpeg_bytes)
    return str(path)
