# Medusa Vision

Implemented with [Keras](https://github.com/keras-team/keras) and [TensorFlow](https://github.com/tensorflow/tensorflow).

Medusa is

1. Based on convolutional neural networks

2. A light-weight computer vision library

2. Facial attribute recognition ( Gender, Emotion, Age ) 

3. Face embedding ( Face encoding ) 

4. Face similarity calculation ( Distance between random face embeddings )

# Installation

- From [PYPI](https://pypi.org/project/medusa-vision/)

    ```shell
    pip install medusa-vision
    ```

- From [.whl](https://github.com/vortezwohl/medusa-vision/releases)

    Download .whl first then run

    ```shell
    pip install ./medusa_vision-X.X.X-py3-none-any.whl
    ```

# Quick Test

```python
from medusa.test import webcam_test
from medusa.model import (
    fer_simple_CNN_param642935_acc66,
    gender_vggface2_VGG16_param134268738_acc97,
    age_vggface2_VGG16_param134674341_acc97
)
webcam_test(
    emo_model=fer_simple_CNN_param642935_acc66,
    gender_model=gender_vggface2_VGG16_param134268738_acc97,
    age_model=age_vggface2_VGG16_param134674341_acc97
)
```

# Quick Guide

- ## Facial attribute recognition

    - Gender detection

        ```python
        from medusa.vision import detect_genders
        PATH_TO_IMAGE = r"./test_images/gosling_and_sister.jpg"
        print(detect_genders(PATH_TO_IMAGE))
        ```

        Input

        ![gosling_and_sister](./test_images//gosling_and_sister.jpg)

        Output

        ```
        (({'gender': 'male', 'probability': 0.9998559}, {'gender': 'female', 'probability': 0.00014410423}), 
        ({'gender': 'female', 'probability': 0.99999905}, {'gender': 'male', 'probability': 9.660015e-07}))
        ```

    - Age detection

        ```python
        from medusa.vision import detect_ages
        PATH_TO_IMAGE = r"./test_images/gosling1.jpg"
        print(detect_ages(PATH_TO_IMAGE))
        ```

        Input

        ![gosling1](./test_images/gosling1.jpg)

        Output

        ```
        (({'age': 23, 'probability': 0.1721634}, {'age': 24, 'probability': 0.17134768}, {'age': 22, 'probability': 0.14511353}, {'age': 26, 'probability': 0.08863783}, {'age': 25, 'probability': 0.08622752}, {'age': 27, 'probability': 0.07673499}, {'age': 28, 'probability': 0.054763753}, {'age': 29, 'probability': 0.05353577}, {'age': 21, 'probability': 0.048544623}, {'age': 20, 'probability': 0.045987632}, {'age': 30, 'probability': 0.03081679}, {'age': 19, 'probability': 0.010067682}, {'age': 31, 'probability': 0.0049784393}, {'age': 32, 'probability': 0.0038630874}, {'age': 18, 'probability': 0.0027914166}, {'age': 33, 'probability': 0.0027323372}, {'age': 17, 'probability': 0.0005507528}, {'age': 35, 'probability': 0.00047534765}, {'age': 34, 'probability': 0.00044520842}, {'age': 36, 'probability': 0.00010379474}, {'age': 37, 'probability': 5.9414804e-05}, {'age': 16, 'probability': 3.5072055e-05}, {'age': 15, 'probability': 1.5215754e-05}, {'age': 14, 'probability': 3.936271e-06}, {'age': 38, 'probability': 1.5921923e-06}, {'age': 39, 'probability': 1.1962729e-06}, {'age': 41, 'probability': 1.0655834e-06}, {'age': 12, 'probability': 2.6780936e-07}, {'age': 42, 'probability': 2.4407265e-07}, {'age': 40, 'probability': 2.3194923e-07}, {'age': 13, 'probability': 2.0135501e-07}, {'age': 11, 'probability': 7.361281e-10}, {'age': 44, 'probability': 5.443918e-10}, {'age': 43, 'probability': 4.2619358e-10}, {'age': 46, 'probability': 4.2352835e-10}, {'age': 45, 'probability': 1.9057463e-10}, {'age': 9, 'probability': 3.4186206e-11}, {'age': 10, 'probability': 2.479897e-11}, {'age': 8, 'probability': 5.336256e-12}, {'age': 47, 'probability': 4.6776025e-12}, {'age': 49, 'probability': 1.1060612e-12}, {'age': 48, 'probability': 1.9979756e-13}, {'age': 7, 'probability': 1.27629146e-14}, {'age': 50, 'probability': 8.4819095e-16}, {'age': 55, 'probability': 4.625277e-16}, {'age': 52, 'probability': 2.2372006e-16}, {'age': 2, 'probability': 1.8421111e-16}, {'age': 51, 'probability': 1.0123701e-17}, {'age': 56, 'probability': 9.344941e-19}, {'age': 5, 'probability': 4.689369e-19}, {'age': 54, 'probability': 3.2003124e-19}, {'age': 53, 'probability': 1.113534e-19}, {'age': 6, 'probability': 4.2698792e-20}, {'age': 58, 'probability': 2.2154548e-20}, {'age': 62, 'probability': 6.4910955e-21}, {'age': 73, 'probability': 1.9908729e-21}, {'age': 59, 'probability': 6.188623e-22}, {'age': 61, 'probability': 6.0775197e-22}, {'age': 63, 'probability': 3.9026103e-22}, {'age': 57, 'probability': 1.1629139e-23}, {'age': 67, 'probability': 1.5029596e-24}, {'age': 60, 'probability': 1.9796683e-25}, {'age': 65, 'probability': 1.3549797e-25}, {'age': 70, 'probability': 6.7419856e-26}, {'age': 69, 'probability': 5.665811e-26}, {'age': 75, 'probability': 1.8290569e-26}, {'age': 68, 'probability': 1.0090035e-26}, {'age': 66, 'probability': 5.605292e-27}, {'age': 64, 'probability': 3.7478e-27}, {'age': 74, 'probability': 5.7158433e-28}, {'age': 4, 'probability': 4.3781928e-29}, {'age': 77, 'probability': 1.3118919e-29}, {'age': 76, 'probability': 3.1017105e-32}, {'age': 80, 'probability': 1.2810593e-33}, {'age': 72, 'probability': 3.2461501e-34}, {'age': 82, 'probability': 3.3592624e-35}, {'age': 71, 'probability': 3.151679e-36}, {'age': 0, 'probability': 7.635638e-37}, {'age': 98, 'probability': 7.413598e-37}, {'age': 85, 'probability': 1.8886916e-37}, {'age': 87, 'probability': 3.4320916e-38}, {'age': 79, 'probability': 2.70847e-38}, {'age': 1, 'probability': 0.0}, {'age': 3, 'probability': 0.0}, {'age': 78, 'probability': 0.0}, {'age': 81, 'probability': 0.0}, {'age': 83, 'probability': 0.0}, {'age': 84, 'probability': 0.0}, {'age': 86, 'probability': 0.0}, {'age': 88, 'probability': 0.0}, {'age': 89, 'probability': 0.0}, {'age': 90, 'probability': 0.0}, {'age': 91, 'probability': 0.0}, {'age': 92, 'probability': 0.0}, {'age': 93, 'probability': 0.0}, {'age': 94, 'probability': 0.0}, {'age': 95, 'probability': 0.0}, {'age': 96, 'probability': 0.0}, {'age': 97, 'probability': 0.0}, {'age': 99, 'probability': 0.0}, {'age': 100, 'probability': 0.0}),)
        ```

    - Emotion detection

        ```python
        from medusa.vision import detect_emos
        PATH_TO_IMAGE = r"./test_images/gosling_happy.jpg"
        print(detect_emos(PATH_TO_IMAGE))
        ```

        Input

        ![gosling_happy](./test_images/gosling_happy.jpg)

        Output

        ```
        (({'emotion': 'happy', 'probability': 0.9441394}, {'emotion': 'surprise', 'probability': 0.04127944}, {'emotion': 'fear', 'probability': 0.0065856194}, {'emotion': 'neutral', 'probability': 0.005740323}, {'emotion': 'angry', 'probability': 0.0020536545}, {'emotion': 'sad', 'probability': 0.0001933232}, {'emotion': 'disgust', 'probability': 8.2883835e-06}),)
        ```

- ## Face Embedding

    - Face Embedding

        1. Demo1

            ```python
            from medusa.embedding import vectorize_face, find_distance, EUCLIDEAN_L2
            from medusa.vision.recognize import match

            PATH_TO_IMAGE1 = r"./test_images/gosling1.jpg"
            PATH_TO_IMAGE2 = r"./test_images/gosling2.jpg"

            face1_embedding = vectorize_face(PATH_TO_IMAGE1)
            face2_embedding = vectorize_face(PATH_TO_IMAGE2)

            distance = find_distance(face1_embedding, face2_embedding, metric=EUCLIDEAN_L2)
            match = match(face1_embedding, face2_embedding, metric=EUCLIDEAN_L2)

            print(f'face1={list(face1_embedding)}\n'
                f'face2={list(face2_embedding)}')
            print(f'distance={distance}')
            print(f'Is the same person? Result={match}')
            ```

            Input

            ![gosling1](./test_images/gosling1.jpg)

            ![gosling2](./test_images/gosling3.jpg)

            Output

            ```
            face1=[0. 0. 0. ... 0. 0. 0.]
            face2=[0. 0. 0. ... 0. 0. 0.]
            distance=0.9430652260780334
            Is the same person? Result=True
            ```

        2. Demo2

            ```python
            from medusa.embedding import vectorize_face, find_distance, EUCLIDEAN_L2
            from medusa.vision.recognize import match

            PATH_TO_IMAGE1 = r"./test_images/gosling1.jpg"
            PATH_TO_IMAGE2 = r"./test_images/bateman1.jpg"

            face1_embedding = vectorize_face(PATH_TO_IMAGE1)
            face2_embedding = vectorize_face(PATH_TO_IMAGE2)

            distance = find_distance(face1_embedding, face2_embedding, metric=EUCLIDEAN_L2)
            match = match(PATH_TO_IMAGE1, PATH_TO_IMAGE2, metric=EUCLIDEAN_L2)

            print(f'face1={face1_embedding}\n'
                f'face2={face2_embedding}')
            print(f'distance={distance}')
            print(f'Is the same person? Result={match}')
            ```

            Input

            ![gosling1](./test_images/gosling1.jpg)

            ![bateman1](./test_images/bateman1.jpg)

            Output

            ```
            face1=[0. 0. 0. ... 0. 0. 0.]
            face2=[0.         0.         0.         ... 0.         0.02576421 0.        ]
            distance=1.2978836297988892
            Is the same person? Result=False
            ```
