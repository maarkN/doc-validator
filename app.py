import json
from flask import Flask, request
from src.face import FacialDocumentValidator
from models.models import ApiArgsModel
from swagger_ui import flask_api_doc
from os import path
from uuid import uuid4

app = Flask(__name__)

flask_api_doc(app, config_path='./config/docs.yaml', url_prefix='/docs', title='API Documentation')

doc_validator = FacialDocumentValidator(
    model='VGG-Face',
    detector='retinaface',
)

@app.route('/')
def index():
    return 'API-ONLINE'

@app.route('/verify-faces', methods=["POST"])
def verify_faces():
    try:
        args = request.get_json()
        ApiArgsModel(**args)
        return json.dumps(doc_validator.compare_face_to_doc(**args))
    except Exception as e:
        return {'api_args_error': str(e)}

@app.route('/verify-faces/image', methods=["POST"])
def verify_faces_with_image():
    try:
        document = request.files['document']
        face = request.files['face']
        print(document)
        print(face)
        doc_image_path = path.join(path.curdir, 'images' , f'{uuid4()}.jpg')
        face_image_path = path.join(path.curdir, 'images' , f'{uuid4()}.jpg')
        document.save(doc_image_path)
        face.save(face_image_path)
        args = {
            'face_img': face_image_path,
            'doc_img': doc_image_path,
            'remove_image': True,
            #'detect_fraud': True, # optional
            #'detect_face_attributes': ["emotion", "age", "gender", "race"], #optional
        }
        ApiArgsModel(**args)
        return json.dumps(doc_validator.compare_face_to_doc(**args))
    except Exception as e:
        return {'api_args_error': str(e)}


if __name__ == '__main__':
    app.run(
        #host="192.168.1.28",
        debug=True)
