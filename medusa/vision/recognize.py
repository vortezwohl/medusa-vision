import keras
import numpy as np

from medusa.model import DEFAULT_EMBEDDING_MODEL
from medusa.embedding import (
    vectorize_face,
    find_thresholds,
    find_distance,
    DEFAULT_METRIC
)


def match(
        face1,
        face2,
        model: keras.Model = DEFAULT_EMBEDDING_MODEL,
        metric: str = DEFAULT_METRIC
) -> bool:
    if not isinstance(face1, np.ndarray):
        face1 = vectorize_face(face1, model)
    if not isinstance(face2, np.ndarray):
        face2 = vectorize_face(face2, model)
    threshold: float = find_thresholds(model)[metric]
    distance: float = find_distance(
        img1=face1,
        img2=face2,
        metric=metric
    )
    return distance <= threshold
