import os
from pathlib import Path
from deepface import DeepFace
from glob import glob
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

DETECTOR_BACKEND = 'opencv'
MODEL_NAME = 'VGG-Face'

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def format_name(path):
    name = Path(path).parent.stem.split()[2:4]
    name[1] = name[1][0] + '.'
    return ' '.join(name)

def get_max_area_face(img_path):
    area = lambda f: f['facial_area']['w'] * f['facial_area']['h']
    faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
    )
    return max(faces, key=area)

def face_permutator(index, docs):
    results = []
    name = format_name(docs[index][0])
    selfie = get_max_area_face(docs[index][1])['face']

    for i, doc in tqdm(enumerate(docs)):
        name_iter = format_name(doc[0])

        doc = get_max_area_face(doc[0])['face']
        
        result = DeepFace.verify(
            img1_path=selfie,
            img2_path=doc,
            model_name='VGG-Face',
            detector_backend='skip',
            enforce_detection=False,
            align=False,
        )
        result['person_1'] = name
        result['person_2'] = name_iter
        results.append(result)

    folder_name = f'{MODEL_NAME}_{DETECTOR_BACKEND}'
    file_name = f'{name.replace(".", "")}.json'
    full_path = f'results/confusion_matrix/{folder_name}/{file_name}'
    os.makedirs(Path(full_path).parent, exist_ok=True)
    with open(f'results/confusion_matrix/{folder_name}/{file_name}', 'w') as f:
         json.dump(results, f, ensure_ascii=True, indent=4)

docs = list(zip(glob('data/*/doc.*'), glob('data/*/selfie.*')))

for index in tqdm(range(len(docs))):
    face_permutator(index=index, docs=docs)