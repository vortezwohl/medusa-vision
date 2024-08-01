import numpy as np
from cv2 import imread, resize


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0  # [0,1]
    if v2:
        x -= 0.5  # [-0.5,0.5]
        x *= 2.0  # [-1,1]
    return x


def read_image(image_name):
        return imread(image_name)


def resize_image(image_array, size):
        return resize(image_array, size)


# one-hot
def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
