from itertools import zip_longest

from cv2 import Mat
import numpy as np
from keras import Model

from medusa.detection.face import detect_faces
from medusa.model import fer_simple_CNN_param642935_acc66, fer_mini_XCEPTION_param58423_acc66
from medusa.model.emotion_detection import EMO
from medusa.model.util.preprocessor import resize_image
from medusa.exception import ModelNotSupportedError


def detect_emo_from_gray(model: Model, gray_face: np.ndarray | Mat) -> tuple:
    if model == fer_simple_CNN_param642935_acc66:
        input_shape = 48, 48
    elif model == fer_mini_XCEPTION_param58423_acc66:
        input_shape = 64, 64
    else:
        raise ModelNotSupportedError(f'Unsupported model: {model}')
    roi_gray = resize_image(
        gray_face,
        input_shape
    )
    predictions = model.predict(roi_gray, verbose=0)[0]
    emotions_detected = {
        k: v
        for k, v in zip_longest(
            EMO,
            predictions,
            fillvalue=0.0
        )
    }
    return tuple([{
                'emotion': emo,
                'probability': val
            } for emo, val in sorted(
                emotions_detected.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
    )


def detect_emos(model: Model, img: any, scale_rate: int = 1.25) -> tuple:
    res = []
    faces = detect_faces(
        img,
        scale_rate
    )
    for face_coordinates, rgb_face, gray_face in faces:
        res.append(
            detect_emo_from_gray(
                model=model,
                gray_face=gray_face,
            )
        )
    return tuple(res)
