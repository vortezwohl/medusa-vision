import os

import cv2
from keras import Model

from medusa.model.emotion_detection import (
    fer_simple_CNN_param642935_acc66,
    fer_mini_XCEPTION_param58423_acc66
)
from medusa.model.gender_detection import imdb_mini_XCEPTION_param52658_acc95
from medusa.detection.face import detect_faces
from medusa.model.util.image_show import draw_text, draw_bounding_box
from medusa.detection.emotion import detect_emo_from_gray
from medusa.detection.gender import detect_gender_from_gray


def run_test(
        emo_model: Model = fer_simple_CNN_param642935_acc66,
        gender_model: Model = imdb_mini_XCEPTION_param52658_acc95,
        color: tuple[int, int, int] = (0, 255, 0),
        ui_thickness: int = 1
):
    if emo_model != fer_simple_CNN_param642935_acc66 and emo_model != fer_mini_XCEPTION_param58423_acc66:
        raise ModuleNotFoundError(f'Unsupported emo model: {emo_model}')
    if gender_model != imdb_mini_XCEPTION_param52658_acc95:
        raise ModuleNotFoundError(f'Unsupported gender model: {gender_model}')
    cap = cv2.VideoCapture(0)
    while True:
        # captures frame and returns boolean value and captured image
        ret, test_img = cap.read()
        if not ret:
            continue
        for face_coordinates, gray_face in detect_faces(test_img):
            draw_text(
                (0, 0, 0, 0),
                test_img,
                "Press 'Q' to quit.",
                color,
                x_offset=int(cv2.CAP_PROP_FRAME_WIDTH + 10),
                y_offset=int(cv2.CAP_PROP_FRAME_HEIGHT + 20),
                font_scale=0.5
            )

            pred_emotions = detect_emo_from_gray(
                model=emo_model,
                gray_face=gray_face
            )

            pred_genders = detect_gender_from_gray(
                model=gender_model,
                gray_face=gray_face
            )

            emo_text = (
                pred_emotions[:2].__str__()
                .replace('(', '')
                .replace(')', '')
                .replace('{', '')
                .replace('}', '')
                .replace("'emotion':", '')
                .replace("'probability':", '')
                .replace("'.", ':')
            )

            gender_text = (
                pred_genders[:2].__str__()
                .replace('(', '')
                .replace(')', '')
                .replace('{', '')
                .replace('}', '')
                .replace("'gender':", '')
                .replace("'probability':", '')
                .replace("'.", ':')
            )

            draw_bounding_box(face_coordinates, test_img, color)
            draw_text(face_coordinates, test_img, emo_text,
                      color, -120, -45, 0.5, ui_thickness)
            draw_text(face_coordinates, test_img, gender_text,
                      color, -120, -20, 0.5, ui_thickness)

        resized_img = cv2.resize(test_img, (1024, 768))
        cv2.imshow('Webcam test', resized_img)

        if cv2.waitKey(5) == ord('q') or cv2.waitKey(10) == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_test()
