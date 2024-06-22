import base64
from dataclasses import dataclass, asdict
import os
from deepface import DeepFace
import cv2
import numpy as np
import pydantic
from typing import Literal
from models.models import ComparatorModel

def show(image, bgr=False):
    if bgr: image = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def base64_to_numpy(base64_string):
    imgdata = base64.b64decode(base64_string)
    return cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)

@dataclass
class FacialDocumentValidator:
    model: str
    detector: str

    def convert_image_to_numpy(self, image):
        if os.path.exists(image):
            return cv2.imread(image)
        imgdata = base64.b64decode(image)
        return cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)

    def extract_faces(self, image, anti_spoofing):
        return DeepFace.extract_faces(
            img_path=image,
            detector_backend=self.detector,
            enforce_detection=False,
            align=True,
            anti_spoofing=anti_spoofing,
        )
    
    def verify_face_and_doc(self, face, doc):
        return DeepFace.verify(
            img1_path=face,
            img2_path=doc,
            model_name='VGG-Face',
            detector_backend='skip',
            enforce_detection=False,
            align=False,
        )
    
    def filter_largest_face(self, extracted_faces):
        if len(extracted_faces) == 1:
            return extracted_faces[0]
        area = lambda f: f['facial_area']['w'] * f['facial_area']['h']
        return max(extracted_faces, key=area)

    def compare_face_to_doc(
            self, face_img, doc_img, 
            detect_fraud=False, 
            detect_face_attributes=False
        ) -> ComparatorModel:

        face_img = self.convert_image_to_numpy(face_img)
        doc_img = self.convert_image_to_numpy(doc_img)

        face_list = self.extract_faces(face_img, anti_spoofing=detect_fraud)
        face = self.filter_largest_face(face_list)

        doc_list = self.extract_faces(doc_img, anti_spoofing=False)
        doc = self.filter_largest_face(doc_list)

        verified = self.verify_face_and_doc(face['face'], doc['face'])
        
        comparator_model = ComparatorModel(
            verified=verified['verified'],
            similarity_distance=verified['distance'],
            similarity_threshold=verified['threshold'],
        )   

        if detect_fraud:
            comparator_model.fake_face = not face['is_real']
            comparator_model.fake_score = 1 - face['antispoof_score']
        if detect_face_attributes:
            attributes = DeepFace.analyze(
                img_path=face_img,
                actions=detect_face_attributes,
                enforce_detection=False,
                detector_backend=self.detector,
                align=True,
            )
            comparator_model.face_attributes = attributes[0]
        
        return comparator_model.to_dict()


# images = list(zip(glob('data/*/selfie.*'), glob('data/*/doc.*')))
# random.shuffle(images)


# for face, doc in images:
#     #for _, doc in images:    
#         f = FacialDocumentValidator(
#             model='VGG-Face',
#             detector='retinaface'
#         )

#         res = f.compare_face_to_doc(
#             face_img=face,
#             doc_img=doc,
#             detect_fraud=False,
#             detect_face_attributes=["emotion", "age", "gender", "race"]
#         )
#         print(json.dumps(res, ensure_ascii=True, indent=4))
#         print()
#         show(cv2.imread(face))
#         show(cv2.imread(doc))