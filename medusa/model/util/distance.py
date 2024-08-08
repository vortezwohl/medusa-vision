import numpy as np


def find_cosine_distance(
    source_embedding: np.ndarray, test_embedding: np.ndarray
) -> np.float64:
    a = np.matmul(np.transpose(source_embedding), test_embedding)
    b = np.sum(np.multiply(source_embedding, source_embedding))
    c = np.sum(np.multiply(test_embedding, test_embedding))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def find_euclidean_distance(
    source_embedding: np.ndarray, test_embedding: np.ndarray
) -> np.float64:
    euclidean_distance = source_embedding - test_embedding
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
