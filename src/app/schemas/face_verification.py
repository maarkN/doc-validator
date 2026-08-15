"""Request/response contracts for the face verification endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FaceAttribute = Literal["emotion", "age", "gender", "race"]


class FaceVerificationRequest(BaseModel):
    """Input for the JSON verification endpoint.

    Images may be a local file path, an http(s) URL or a base64-encoded
    string (with or without a ``data:image/...;base64,`` prefix).
    """

    model_config = ConfigDict(extra="forbid")

    face_img: str = Field(description="Selfie image: path, URL or base64.")
    doc_img: str = Field(description="Document image: path, URL or base64.")
    detect_fraud: bool = Field(
        default=False,
        description="Run anti-spoofing on the selfie and report fake_face/fake_score.",
    )
    detect_face_attributes: list[FaceAttribute] | None = Field(
        default=None,
        description="Facial attributes to analyze on the selfie.",
    )
    remove_image: bool = Field(
        default=False,
        description="Delete local input files after verification (ignored for "
        "URL and base64 inputs).",
    )


class FaceVerificationResult(BaseModel):
    """Outcome of comparing a selfie against a document photo."""

    verified: bool = Field(description="Whether both images show the same person.")
    similarity_distance: float = Field(
        description="Embedding distance between the two faces (lower is closer)."
    )
    similarity_threshold: float = Field(
        description="Distance threshold below which faces are considered a match."
    )
    fake_face: bool | None = Field(
        default=None, description="Anti-spoofing verdict; present when requested."
    )
    fake_score: float | None = Field(
        default=None, description="Spoof likelihood in [0, 1]; present when requested."
    )
    face_attributes: dict[str, Any] | None = Field(
        default=None, description="Facial attributes; present when requested."
    )
