from keras import Model
from keras._tf_keras.keras.layers import Flatten

from medusa.model import vggface2_VGG16_param52658_acc97

THRESHOLDS = {
    'VGG-Face-Embedding': {
        'cosine': 0.68,
        'euclidean': 1.17,
        'euclidean_l2': 1.17
    }
}

embedding_vggface2_VGG16_param52658_acc97 = Model(
    inputs=vggface2_VGG16_param52658_acc97.inputs,
    outputs=Flatten()(
        vggface2_VGG16_param52658_acc97
        .layers[-5]
        .output
    )
)
