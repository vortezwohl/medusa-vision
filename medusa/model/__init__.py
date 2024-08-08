import keras

keras.layers.deserialize._kerastypes = {
    'Sequential': keras.Sequential
}

from .cnn import create_simple_CNN, create_mini_XCEPTION, create_VGG16
from .gender_detection import (
    imdb_mini_XCEPTION_param52658_acc95,
    gender_vggface2_VGG16_param134268738_acc97
)
from .emotion_detection import (
    fer_simple_CNN_param642935_acc66,
    fer_mini_XCEPTION_param58423_acc66
)
from .age_detection import age_vggface2_VGG16_param134674341_acc97
from .vgg16_face import vggface2_VGG16_param52658_acc97
from .face_embedding import embedding_vggface2_VGG16_param52658_acc97

DEFAULT_EMO_MODEL = fer_simple_CNN_param642935_acc66
DEFAULT_GENDER_MODEL = gender_vggface2_VGG16_param134268738_acc97
DEFAULT_AGE_MODEL = age_vggface2_VGG16_param134674341_acc97
DEFAULT_EMBEDDING_MODEL = embedding_vggface2_VGG16_param52658_acc97
