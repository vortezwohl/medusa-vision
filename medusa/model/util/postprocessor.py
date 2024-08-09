import numpy as np
import cv2
import keras

from medusa.model.util.preprocessor import resize_image
from medusa.model.util.distance import (
    find_cosine_distance,
    find_euclidean_distance
)
from medusa.exception import (
    ModelNotSupportedError,
    MetricNotSupportedError,
    MoreThanOneFaceError
)
from medusa.model import (
    embedding_vggface2_VGG16_param52658_acc97,
    DEFAULT_EMBEDDING_MODEL
)
from medusa.model.face_embedding import (
    COSINE,
    EUCLIDEAN,
    EUCLIDEAN_L2,
    DEFAULT_METRIC
)
from medusa.model.face_embedding import THRESHOLDS
from medusa.vision import detect_faces


# euclidean normalization: x / ||x||^2
def l2_normalize(x: np.ndarray | list) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def vectorize_face_from_ndarray(
        img: np.ndarray | cv2.Mat,
        model: keras.Model = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    if model == embedding_vggface2_VGG16_param52658_acc97:
        return l2_normalize(
            embedding_vggface2_VGG16_param52658_acc97.predict(
                img,
                verbose=False
            )[0]
        )
    raise ModelNotSupportedError(f'Unsupported model: {model}')


# only one face should be provided
def vectorize_face(
        img,
        model: keras.Model = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    faces_detected = detect_faces(img)
    if len(faces_detected) != 1:
        raise MoreThanOneFaceError(f'More than one face detected. Detected faces: {len(faces_detected)}')
    cord, rgb, gray = faces_detected[0]
    if model == embedding_vggface2_VGG16_param52658_acc97:
        resized_input = resize_image(rgb, (224, 224))
    else:
        raise ModelNotSupportedError(f'Unsupported model: {model}')
    return vectorize_face_from_ndarray(
        resized_input,
        model
    )


def find_thresholds(model: keras.Model = DEFAULT_EMBEDDING_MODEL) -> dict:
    if model == embedding_vggface2_VGG16_param52658_acc97:
        return THRESHOLDS['VGG-Face-Embedding']
    else:
        raise ModelNotSupportedError(f'Unsupported model: {model}')


def find_distance(
        img1: np.ndarray,
        img2: np.ndarray,
        metric: str = DEFAULT_METRIC
) -> np.float64:
    if metric == COSINE:
        return find_cosine_distance(img1, img2)
    elif metric == EUCLIDEAN:
        return find_euclidean_distance(img1, img2)
    elif metric == EUCLIDEAN_L2:
        return find_euclidean_distance(
            l2_normalize(img1), l2_normalize(img2)
        )
    else:
        raise MetricNotSupportedError(f'Unsupported metric: {metric}')


def output_stringify(output: tuple, tag: str, top_k: int) -> str:
    return (
        output[:top_k].__str__()
        .replace('(', '')
        .replace(')', '')
        .replace('{', '')
        .replace('}', '')
        .replace(f"'{tag}':", '')
        .replace("'probability':", '')
        .replace("',", ':')
        .replace("'", '')
        .replace(' ', '')
        .replace(',', ' ')
    )
