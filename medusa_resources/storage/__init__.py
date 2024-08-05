import os

import medusa_resources

MEDUSA_RESOURCES_URL = 'https://github.com/vortezwohl/medusa-vision-weights/releases/download/resources/'
MEDUSA_RESOURCES_GITHUB = 'https://github.com/vortezwohl/medusa-vision-weights/releases/tag/resources'
WEIGHTS: list = [
    'fer2013_mini_XCEPTION_param58423_acc0.66.hdf5',
    'imdb_mini_XCEPTION_param52658_acc0.95.hdf5',
    'fer2013_simple_CNN_param642935_acc0.66.hdf5',
    'gender_vggface2_VGG16_param134268738_acc0.97.h5',
    'age_vggface2_VGG16_param134674341_acc0.97.h5',
    'vggface2_VGG16_param145002878_acc0.97.h5'
]
OPENCV: list = [
    'haarcascade_frontalface_default.xml'
]

if not medusa_resources.util.file_exists(medusa_resources.MEDUSA_STORAGE_WEIGHTS):
    os.mkdir(medusa_resources.MEDUSA_STORAGE_WEIGHTS)

if not medusa_resources.util.file_exists(medusa_resources.MEDUSA_STORAGE_OPENCV):
    os.mkdir(medusa_resources.MEDUSA_STORAGE_OPENCV)
