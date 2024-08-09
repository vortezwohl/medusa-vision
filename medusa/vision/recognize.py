import keras

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
    face1_embedding = vectorize_face(face1, model)
    face2_embedding = vectorize_face(face2, model)
    threshold: float = find_thresholds(model)[metric]
    distance: float = find_distance(
        img1=face1_embedding,
        img2=face2_embedding,
        metric=metric
    )
    return distance <= threshold
