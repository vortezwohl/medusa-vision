from itertools import zip_longest

from cv2 import Mat
import numpy as np
from keras import Model
from keras._tf_keras.keras.preprocessing import image as tfimage

from medusa.detection.face import detect_faces
from medusa.model.emotion_detection import EMO, fer_simple_CNN_param642935_acc66, fer_mini_XCEPTION_param58423_acc66
from medusa.model.util.preprocessor import resize_image


def detect_emo_from_gray(model: Model, gray_face: np.ndarray | Mat) -> tuple | None:
    if model == fer_simple_CNN_param642935_acc66:
        input_shape = 48, 48
    elif model == fer_mini_XCEPTION_param58423_acc66:
        input_shape = 64, 64
    else:
        raise ModuleNotFoundError(f'Unsupported model: {model}')
    roi_gray = resize_image(
        gray_face,
        input_shape
    )
    img_pixels = tfimage.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    predictions = model.predict(img_pixels, verbose=0)[0]
    emotions_detected = {
        k: v
        for k, v in zip_longest(
            EMO,
            predictions,
            fillvalue=0.0
        )
    }
    sorted_items_desc = tuple([{
                'emotion': emo,
                'probability': val
            } for emo, val in sorted(
                emotions_detected.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
    )
    return sorted_items_desc


def detect_emos(model: Model, img: any, scale_rate: int = 1.25) -> tuple:
    res = []
    faces = detect_faces(
        img,
        scale_rate
    )
    for face_coordinates, gray_face in faces:
        res.append(
            detect_emo_from_gray(
                model=model,
                gray_face=gray_face,
            )
        )
    return tuple(res)
