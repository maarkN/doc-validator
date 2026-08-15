"""Boundary with the DeepFace library.

Every call here is blocking (TensorFlow/PyTorch inference), so callers must
run outside the event loop — the API layer uses plain ``def`` endpoints.
DeepFace is imported lazily: it is an optional heavy dependency installed
only where inference runs, and tests replace it at the module boundary.
"""

from typing import Any

import numpy as np


class DeepFaceEngine:
    """Thin, parameter-preserving wrapper around the DeepFace entry points."""

    def __init__(self, recognition_model: str, detector_backend: str) -> None:
        """Store the model and detector used by all calls.

        Args:
            recognition_model: DeepFace recognition model (e.g. ``VGG-Face``).
            detector_backend: DeepFace detector (e.g. ``retinaface``).
        """
        self.recognition_model = recognition_model
        self.detector_backend = detector_backend

    def extract_faces(
        self, image: np.ndarray, anti_spoofing: bool
    ) -> list[dict[str, Any]]:
        """Detect and align every face in ``image``."""
        from deepface import DeepFace

        return DeepFace.extract_faces(  # type: ignore[no-any-return]
            img_path=image,
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=True,
            anti_spoofing=anti_spoofing,
        )

    def verify(self, face_crop: np.ndarray, doc_crop: np.ndarray) -> dict[str, Any]:
        """Compare two pre-extracted face crops.

        Detection is skipped and alignment disabled because the crops were
        already aligned by :meth:`extract_faces`; changing these parameters
        changes the distance numbers.
        """
        from deepface import DeepFace

        return DeepFace.verify(  # type: ignore[no-any-return]
            img1_path=face_crop,
            img2_path=doc_crop,
            model_name=self.recognition_model,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )

    def analyze(self, image: np.ndarray, actions: list[str]) -> list[dict[str, Any]]:
        """Estimate facial attributes for the faces found in ``image``."""
        from deepface import DeepFace

        return DeepFace.analyze(  # type: ignore[no-any-return]
            img_path=image,
            actions=actions,
            enforce_detection=False,
            detector_backend=self.detector_backend,
            align=True,
        )
