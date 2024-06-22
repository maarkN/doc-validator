import timeit
from deepface import DeepFace
from glob import glob
import cv2
import numpy as np

def show(image, bgr=False):
    if bgr:
        image = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

docs = list(zip(glob('data/*/doc.*'), glob('data/*/selfie.*')))

detector_backend = 'opencv'

for doc in docs:
    t0 = timeit.default_timer()
    selfie_face = DeepFace.extract_faces(
        img_path=doc[1],
        detector_backend=detector_backend,
        enforce_detection=False,
    )

    doc_face = DeepFace.extract_faces(
        img_path=doc[0],
        detector_backend=detector_backend,
        enforce_detection=False,
    )

    result = DeepFace.verify(
        img1_path=selfie_face[0]['face'],
        img2_path=doc_face[0]['face'],
        model_name='VGG-Face',
        detector_backend='skip',
        enforce_detection=False,
        align=True,
        # anti_spoofing=True,
    )
    print(result)
    t1 = timeit.default_timer()
    print(f'[{t1-t0:.4f}]')