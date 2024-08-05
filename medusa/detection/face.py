import cv2
import numpy as np
from cv2 import Mat

from medusa.exception import UnsupportedFormatError
from medusa_resources.util import file_exists
from medusa_resources import MEDUSA_STORAGE_OPENCV

HAAR_CASCADE_CONFIG = f'{MEDUSA_STORAGE_OPENCV}\\haarcascade_frontalface_default.xml'

face_haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_CONFIG)


# (x, y, w, h)
def detect_faces(img, scale_rate: int = 1.25) -> list:
    res = []
    if isinstance(img, str):
        if not file_exists(img):
            raise FileNotFoundError(f'Image: {img} not found')
        original_image = cv2.imread(img)
    elif isinstance(img, np.ndarray | Mat):
        original_image = img
    else:
        try:
            # img = BytesIO(image_data_from_internet)
            original_image = cv2.imdecode(
                np.frombuffer(
                    img.read(),
                    np.uint8
                ),
                cv2.IMREAD_COLOR
            )
        except:
            raise UnsupportedFormatError(f'Unsupported input format: {type(img)}')
    gray_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2GRAY
    )
    faces_detected = face_haar_cascade.detectMultiScale(
        gray_image,
        scale_rate,
        5
    )
    for x, y, w, h in faces_detected:
        res.append((
                (x, y, w, h),
                original_image[y:y + h, x:x + w],
                gray_image[y:y + h, x:x + w]
            )
        )
    return res


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off
