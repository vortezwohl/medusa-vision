import cv2
import keras._tf_keras.keras.preprocessing

from medusa.exception import UnsupportedFormatError
from medusa.model.util.distance import *
from medusa_resources.util import file_exists


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0  # [0,1]
    if v2:
        x -= 0.5  # [-0.5,0.5]
        x *= 2.0  # [-1,1]
    return x


def read_image(image_name):
    if isinstance(image_name, str):
        if not file_exists(image_name):
            raise FileNotFoundError(f'Image: {image_name} not found')
        original_image = cv2.imread(image_name)
    elif isinstance(image_name, np.ndarray | cv2.Mat):
        original_image = image_name
    else:
        try:
            # img = BytesIO(image_data_from_internet)
            original_image = cv2.imdecode(
                np.frombuffer(
                    image_name.read(),
                    np.uint8
                ),
                cv2.IMREAD_COLOR
            )
        except Exception:
            raise UnsupportedFormatError(f'Unsupported input format: {type(image_name)}')
    return original_image


def resize_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)
    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    # make it 4-dimensional how ML models expect
    img = keras._tf_keras.keras.preprocessing.image.img_to_array(img)
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
