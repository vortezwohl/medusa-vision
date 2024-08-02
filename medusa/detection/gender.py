from itertools import zip_longest

from cv2 import Mat
import numpy as np
from keras import Model
from keras._tf_keras.keras.preprocessing import image as tfimage

from medusa.detection.face import detect_faces
from medusa.exception import ModelNotSupportedError
from medusa.model import imdb_mini_XCEPTION_param52658_acc95
from medusa.model.gender_detection import GENDER
from medusa.model.util.preprocessor import resize_image


def detect_gender_from_gray(model: Model, gray_face: np.ndarray | Mat) -> tuple | None:
    if model != imdb_mini_XCEPTION_param52658_acc95:
        raise ModelNotSupportedError(f'Unsupported gender model: {model}')
    roi_gray = resize_image(
        gray_face,
        (64, 64)
    )
    img_pixels = tfimage.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    predictions = model.predict(img_pixels, verbose=0)[0]
    emotions_detected = {
        k: v
        for k, v in zip_longest(
            GENDER,
            predictions,
            fillvalue=0.0
        )
    }
    sorted_items_desc = tuple([{
                'gender': emo,
                'probability': val
            } for emo, val in sorted(
                emotions_detected.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
    )
    return sorted_items_desc


def detect_genders(model: Model, img: any, scale_rate: int = 1.25) -> tuple:
    res = []
    faces = detect_faces(
        img,
        scale_rate
    )
    for face_coordinates, gray_face in faces:
        res.append(
            detect_gender_from_gray(
                model=model,
                gray_face=gray_face,
            )
        )
    return tuple(res)
