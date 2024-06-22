import json
from flask import Flask, request
from src.face import FacialDocumentValidator
from models.models import ApiArgsModel

app = Flask(__name__)

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


if __name__ == '__main__':
    app.run(
        #host="192.168.1.28",
        debug=True)
