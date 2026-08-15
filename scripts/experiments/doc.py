from deepface import DeepFace
from glob import glob
import cv2
from PIL import Image
import numpy as np

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def largest_area(faces):
    area = lambda f: f['facial_area']['w'] * f['facial_area']['h']
    return max(faces, key=area)

def face_detector(docs):
    for doc in docs:
        dfs = DeepFace.extract_faces(
            img_path=doc[1],
            detector_backend='retinaface',
        )
        show(cv2.imread(doc[1]))
        print(dfs)

        max_face_area = largest_area(dfs)
        show(cv2.cvtColor((max_face_area['face']*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

def face_comparator(docs):
    for doc in docs:
        # detector backend = skip
        result = DeepFace.verify(
            img1_path=doc[0],
            img2_path=doc[1],
            model_name='VGG-Face',
            detector_backend='retinaface',
            enforce_detection=False,
        )
        d = {}
        d['v'] = result['verified']
        d['d'] = result['distance']
        

docs = list(zip(glob('data/*/doc.*'), glob('data/*/selfie.*')))

face_detector(docs)
