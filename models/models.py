import pydantic
from typing import Literal
from dataclasses import dataclass, asdict

class ApiArgsModel(pydantic.BaseModel):
    face_img: str
    doc_img: str
    detect_fraud: bool = None
    detect_face_attributes: list[Literal["emotion", "age", "gender", "race"]] = None
    remove_image: bool = None

@dataclass
class ComparatorModel:
    verified: bool
    similarity_distance: float
    similarity_threshold: float
    fake_face: bool | None = None
    fake_score: float | None = None
    face_attributes: dict | None = None

    def to_dict(self):
        return asdict(self)
    