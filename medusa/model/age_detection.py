from medusa.model import create_VGG16
from medusa_resources import MEDUSA_STORAGE_WEIGHTS

AGE_VGGFACE2_VGG16_WEIGHT = f'{MEDUSA_STORAGE_WEIGHTS}\\age_vggface2_VGG16_param134674341_acc0.97.h5'

AGE = tuple([i for i in range(101)])

age_vggface2_VGG16_param134674341_acc97 = create_VGG16(
    input_shape=(224, 224, 3),
    num_classes=len(AGE)
)
age_vggface2_VGG16_param134674341_acc97.load_weights(AGE_VGGFACE2_VGG16_WEIGHT)
