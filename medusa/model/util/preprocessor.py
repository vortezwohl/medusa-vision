from cv2 import imread, resize, Mat
from keras._tf_keras.keras.preprocessing import image
from keras import Model

from medusa.exception import ModelNotSupportedError, MetricNotSupportedError, MoreThanOneFaceError
from medusa.model import embedding_vggface2_VGG16_param52658_acc97
from medusa.model.face_embedding import THRESHOLDS
from medusa.model.util.distance import *
from medusa.vision import detect_faces


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0  # [0,1]
    if v2:
        x -= 0.5  # [-0.5,0.5]
        x *= 2.0  # [-1,1]
    return x


def read_image(image_name):
    return imread(image_name)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off


def resize_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = resize(img, dsize)
    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = resize(img, target_size)
    # make it 4-dimensional how ML models expect
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)
    return img


# one-hot
def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical


# euclidean normalization: x / ||x||^2
def l2_normalize(x: np.ndarray | list) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def vectorize_face_from_ndarray(model: Model, img: np.ndarray | Mat) -> np.ndarray:
    if model == embedding_vggface2_VGG16_param52658_acc97:
        return l2_normalize(embedding_vggface2_VGG16_param52658_acc97.predict(img, verbose=False))[0]
    raise ModelNotSupportedError(f'Unsupported model: {model}')


# only one face should be provided
def vectorize_face(model: Model, img) -> np.ndarray:
    faces_detected = detect_faces(img)
    if len(faces_detected) != 1:
        raise MoreThanOneFaceError(f'More than one face detected. Detected faces: {len(faces_detected)}')
    cord, rgb, gray = faces_detected[0]
    if model == embedding_vggface2_VGG16_param52658_acc97:
        resized_input = resize_image(rgb, (224, 224))
    else:
        raise ModelNotSupportedError(f'Unsupported model: {model}')
    return vectorize_face_from_ndarray(
        model,
        resized_input
    )


def find_thresholds(model: Model) -> dict:
    if model == embedding_vggface2_VGG16_param52658_acc97:
        return THRESHOLDS['VGG-Face-Embedding']
    else:
        raise ModelNotSupportedError(f'Unsupported model: {model}')


def find_distance(
        img1: np.ndarray,
        img2: np.ndarray,
        metric: str = 'euclidean_l2'
) -> np.float64:
    if metric == 'cosine':
        return find_cosine_distance(img1, img2)
    elif metric == 'euclidean':
        return find_euclidean_distance(img1, img2)
    elif metric == 'euclidean_l2':
        return find_euclidean_distance(
            l2_normalize(img1), l2_normalize(img2)
        )
    else:
        raise MetricNotSupportedError(f'Unsupported metric: {metric}')

