from itertools import zip_longest

from cv2 import Mat
import numpy as np
from keras import Model

from medusa.detection.face import detect_faces
from medusa.exception import ModelNotSupportedError
from medusa.model import age_vggface2_VGG16_param134674341_acc97
from medusa.model.age_detection import AGE
from medusa.model.util.preprocessor import resize_image


def detect_age_from_rgb(model: Model, rgb_face: np.ndarray | Mat) -> tuple:
    if model != age_vggface2_VGG16_param134674341_acc97:
        raise ModelNotSupportedError(f'Unsupported gender model: {model}')
    roi_rgb = resize_image(
        rgb_face,
        (224, 224)
    )
    predictions = model.predict(roi_rgb, verbose=0)[0]
    ages_detected = {
        k: v
        for k, v in zip_longest(
            AGE,
            predictions,
            fillvalue=0.0
        )
    }
    return tuple([{
                'age': age,
                'probability': val
            } for age, val in sorted(
                ages_detected.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
    )


def detect_ages(model: Model, img: any, scale_rate: int = 1.25) -> tuple:
    res = []
    faces = detect_faces(
        img,
        scale_rate
    )
    for face_coordinates, rgb_face, gray_face in faces:
        res.append(
            detect_age_from_rgb(
                model=model,
                rgb_face=rgb_face
            )
        )
    return tuple(res)
