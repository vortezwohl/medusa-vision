import os
import cv2

from medusa.model.util.preprocessor import read_image
from medusa_resources import MEDUSA_STORAGE_OPENCV
from medusa_resources.util import file_exists

HAAR_CASCADE_CONFIG = f'{MEDUSA_STORAGE_OPENCV}\\haarcascade_frontalface_default.xml'

face_haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_CONFIG)


# (x, y, w, h)
def detect_faces(img, scale_rate: int = 1.25, offsets: tuple = (0, 0)) -> list | tuple:
    res = []
    original_image = read_image(img)
    gray_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2GRAY
    )
    faces_detected = face_haar_cascade.detectMultiScale(
        gray_image,
        scale_rate,
        5
    )
    for coordinates in faces_detected:
        x, y, w, h = apply_offsets(coordinates, offsets)
        res.append((
                (x, y, w, h),
                original_image[y:y + h, x:x + w],
                gray_image[y:y + h, x:x + w]
            )
        )
    return res


def count_faces(img, scale_rate: int = 1.25):
    return len(detect_faces(img, scale_rate))


def apply_offsets(face_coordinates, offsets) -> tuple[int, int, int, int]:
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, y - y_off, width + x_off, height + y_off


def split_and_export_faces(img: str, export_dir: str, scale_rate: int = 1.25, offsets: tuple = (15, 60)):
    if not file_exists(export_dir):
        os.makedirs(export_dir)
    export_dir = export_dir.replace('/', '').replace('\\', '').replace('.', '')
    for index, (coordinates, rgb, gray) in enumerate(detect_faces(img, scale_rate, offsets)):
        cv2.imwrite(
            filename=f"{export_dir}/face{index}.png",
            img=rgb,
            params=[int(cv2.IMWRITE_PNG_COMPRESSION), 1]
        )
