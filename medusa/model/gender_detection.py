import os

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import deserialize

from medusa.model.cnn import create_mini_XCEPTION

deserialize._kerastypes = {
    'Sequential': Sequential
}

PATH = os.path.dirname(__file__)

IMDB_MINI_XCEPTION_WEIGHT = f'{PATH}\\weight\\imdb_mini_XCEPTION_param52658_epoch21_acc0.95.hdf5'

GENDER = ('female', 'male')

imdb_mini_XCEPTION_param52658_acc95 = create_mini_XCEPTION((64, 64, 1), 2)
imdb_mini_XCEPTION_param52658_acc95.load_weights(IMDB_MINI_XCEPTION_WEIGHT)
