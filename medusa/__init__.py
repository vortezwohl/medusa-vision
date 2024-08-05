__version__ = '0.5.3'
__author__ = 'vortezwohl'
__email__ = 'vortezwohl@proton.me'

from medusa_resources.storage.storage_manager import install_resources
from medusa_resources.exception import ResourceDownloadError
from medusa_resources.storage import MEDUSA_RESOURCES_GITHUB

if install_resources(rollback_retries=3):

    import keras
    from .model import *
    from .detection import *
    from .test import webcam_test

    keras.layers.deserialize._kerastypes = {
        'Sequential': keras.Sequential
    }

    DEFAULT_EMO_MODEL = fer_simple_CNN_param642935_acc66
    DEFAULT_GENDER_MODEL = gender_vggface2_VGG16_param134268738_acc97
    DEFAULT_AGE_MODEL = age_vggface2_VGG16_param134674341_acc97

else:
    raise ResourceDownloadError(f'Failed Downloading resources. Please try again or install manually from {MEDUSA_RESOURCES_GITHUB}.')
