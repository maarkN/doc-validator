import base64
import json
from pathlib import Path
import cv2
import requests
import random
from glob import glob
from src.face import FacialDocumentValidator

url = 'http://127.0.0.1:5000/verify-faces'

images = list(zip(glob('data/*/selfie.*'), glob('data/*/doc.*')))
random.shuffle(images)

def to_base64(image):
    _, buffer = cv2.imencode('.jpg', cv2.imread(image))
    return base64.b64encode(buffer).decode('utf-8')

for face, doc in images:
    face_base64 = to_base64(face)
    doc_base64 = to_base64(doc)
    params = {
        'face_img': face_base64,
        'doc_img': doc_base64,
        #'detect_fraud': True, # optional
        #'detect_face_attributes': ["emotion", "age", "gender", "race"], #optional
    }

    response = requests.post(url, json=params)
    if response.status_code == 200:
        print("Success:\n", json.dumps(json.loads(response.content), indent=4, ensure_ascii=False))
    else:
        print("Error:", response.status_code, response.json())
    print()