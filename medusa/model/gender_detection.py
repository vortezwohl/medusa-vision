from medusa.model import create_VGG16, create_mini_XCEPTION
from medusa_resources import MEDUSA_STORAGE_WEIGHTS

IMDB_MINI_XCEPTION_WEIGHT = f'{MEDUSA_STORAGE_WEIGHTS}\\imdb_mini_XCEPTION_param52658_acc0.95.hdf5'
GENDER_VGGFACE2_VGG16_WEIGHT = f'{MEDUSA_STORAGE_WEIGHTS}\\gender_vggface2_VGG16_param134268738_acc0.97.h5'

GENDER = ('female', 'male')

imdb_mini_XCEPTION_param52658_acc95 = create_mini_XCEPTION((64, 64, 1), 2)
imdb_mini_XCEPTION_param52658_acc95.load_weights(IMDB_MINI_XCEPTION_WEIGHT)

gender_vggface2_VGG16_param134268738_acc97 = create_VGG16(
    input_shape=(224, 224, 3),
    num_classes=len(GENDER)
)
gender_vggface2_VGG16_param134268738_acc97.load_weights(GENDER_VGGFACE2_VGG16_WEIGHT)
