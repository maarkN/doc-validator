"""Characterization-test fixtures for the legacy Flask app.

DeepFace is stubbed at the process boundary (sys.modules) BEFORE the app is
imported, because importing the real deepface pulls TensorFlow and downloads
model weights. Everything else (Flask, pydantic, cv2, numpy) is real, so these
tests freeze the observable HTTP contract of the legacy application layer:
routing, validation, error envelope, serialization and DeepFace call
orchestration.
"""

import base64
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class DeepFaceStub:
    """Records calls and returns canned, controllable responses."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.extract_faces_calls = []
        self.verify_calls = []
        self.analyze_calls = []
        # One face per image by default; tests may append more.
        self.faces = [
            {
                "face": np.zeros((10, 10, 3), dtype=np.float64),
                "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
                "confidence": 0.99,
                "is_real": True,
                "antispoof_score": 0.9,
            }
        ]
        self.verify_result = {
            "verified": True,
            "distance": 0.42,
            "threshold": 0.68,
        }
        self.analyze_result = [{"age": 30, "dominant_emotion": "happy"}]

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

# The legacy app resolves config/docs.yaml and images/ relative to the CWD.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from app import app as legacy_app  # noqa: E402


@pytest.fixture()
def deepface_stub():
    _stub.reset()
    return _stub


@pytest.fixture()
def client(deepface_stub):
    legacy_app.config["TESTING"] = True
    with legacy_app.test_client() as test_client:
        yield test_client


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
