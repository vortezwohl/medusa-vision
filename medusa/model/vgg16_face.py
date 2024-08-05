from medusa.model import create_VGG16
from medusa_resources import MEDUSA_STORAGE_WEIGHTS

VGGFACE2_VGG16_WEIGHT = f'{MEDUSA_STORAGE_WEIGHTS}\\vggface2_VGG16_param145002878_acc0.97.h5'

vggface2_VGG16_param52658_acc97 = create_VGG16(input_shape=(224, 224, 3))
vggface2_VGG16_param52658_acc97.load_weights(VGGFACE2_VGG16_WEIGHT)
