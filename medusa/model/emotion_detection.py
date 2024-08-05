from medusa.model import create_simple_CNN, create_mini_XCEPTION
from medusa_resources import MEDUSA_STORAGE_WEIGHTS

FER_SIMPLE_CNN_WEIGHT = f'{MEDUSA_STORAGE_WEIGHTS}\\fer2013_simple_CNN_param642935_acc0.66.hdf5'
FER_MINI_XCEPTION_WEIGHT = f'{MEDUSA_STORAGE_WEIGHTS}\\fer2013_mini_XCEPTION_param58423_acc0.66.hdf5'

EMO = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

fer_simple_CNN_param642935_acc66 = create_simple_CNN((48, 48, 1), 7)
fer_simple_CNN_param642935_acc66.load_weights(FER_SIMPLE_CNN_WEIGHT)

fer_mini_XCEPTION_param58423_acc66 = create_mini_XCEPTION((64, 64, 1), 7)
fer_mini_XCEPTION_param58423_acc66.load_weights(FER_MINI_XCEPTION_WEIGHT)
