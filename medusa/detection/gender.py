from itertools import zip_longest

from cv2 import Mat
import numpy as np
from keras import Model

from medusa.detection.face import detect_faces
from medusa.exception import ModelNotSupportedError
from medusa.model import imdb_mini_XCEPTION_param52658_acc95, gender_vggface2_VGG16_param134268738_acc97
from medusa.model.gender_detection import GENDER
from medusa.model.util.preprocessor import resize_image


def detect_gender_from_gray(model: Model, gray_face: np.ndarray | Mat) -> tuple:
    if model != imdb_mini_XCEPTION_param52658_acc95:
        raise ModelNotSupportedError(f'Unsupported gender model: {model}')
    roi_gray = resize_image(
        gray_face,
        (64, 64)
    )
    predictions = model.predict(roi_gray, verbose=0)[0]
    genders_detected = {
        k: v
        for k, v in zip_longest(
            GENDER,
            predictions,
            fillvalue=0.0
        )
    }
    return tuple([{
                'gender': gender,
                'probability': val
            } for gender, val in sorted(
                genders_detected.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
    )


def detect_gender_from_rgb(model: Model, rgb_face: np.ndarray | Mat) -> tuple:
    if model != gender_vggface2_VGG16_param134268738_acc97:
        raise ModelNotSupportedError(f'Unsupported gender model: {model}')
    roi_rgb = resize_image(
        rgb_face,
        (224, 224)
    )
    predictions = model.predict(roi_rgb, verbose=0)[0]
    genders_detected = {
        k: v
        for k, v in zip_longest(
            GENDER,
            predictions,
            fillvalue=0.0
        )
    }
    return tuple([{
                'gender': gender,
                'probability': val
            } for gender, val in sorted(
                genders_detected.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
    )


def detect_genders(model: Model, img: any, scale_rate: int = 1.25) -> tuple:
    res = []
    faces = detect_faces(
        img,
        scale_rate
    )
    for face_coordinates, rgb_face, gray_face in faces:
        if model == imdb_mini_XCEPTION_param52658_acc95:
            res.append(
                detect_gender_from_gray(
                    model=model,
                    gray_face=gray_face
                )
            )
        elif model == gender_vggface2_VGG16_param134268738_acc97:
            res.append(
                detect_gender_from_rgb(
                    model=model,
                    rgb_face=rgb_face
                )
            )
    return tuple(res)
