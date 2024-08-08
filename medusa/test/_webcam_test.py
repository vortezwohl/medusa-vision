import cv2
from keras import Model

from medusa.model import *
from medusa.vision import *
from medusa.model.util.image_show import draw_text, draw_bounding_box
from medusa.exception import ModelNotSupportedError
from medusa.model.util.postprocessor import output_stringify


def run_test(
        emo_model: Model = fer_simple_CNN_param642935_acc66,
        gender_model: Model = gender_vggface2_VGG16_param134268738_acc97,
        age_model: Model = age_vggface2_VGG16_param134674341_acc97,
        color: tuple[int, int, int] = (0, 255, 0),
        ui_thickness: int = 1
):
    if emo_model != fer_simple_CNN_param642935_acc66 and emo_model != fer_mini_XCEPTION_param58423_acc66:
        raise ModelNotSupportedError(f'Unsupported emo model: {emo_model}')
    if gender_model != gender_vggface2_VGG16_param134268738_acc97 and gender_model != imdb_mini_XCEPTION_param52658_acc95:
        raise ModelNotSupportedError(f'Unsupported gender model: {gender_model}')
    if age_model != age_vggface2_VGG16_param134674341_acc97:
        raise ModelNotSupportedError(f'Unsupported gender model: {age_model}')
    cap = cv2.VideoCapture(0)
    while True:
        # captures frame and returns boolean value and captured image
        ret, test_img = cap.read()
        if not ret:
            continue
        for face_coordinates, rgb_face, gray_face in detect_faces(test_img):
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
            predict_ages = detect_age_from_rgb(
                model=age_model,
                rgb_face=rgb_face
            )
            pred_genders: tuple = tuple()
            if gender_model == gender_vggface2_VGG16_param134268738_acc97:
                pred_genders = detect_gender_from_rgb(
                    model=gender_model,
                    rgb_face=rgb_face
                )
            elif gender_model == imdb_mini_XCEPTION_param52658_acc95:
                pred_genders = detect_gender_from_gray(
                    model=gender_model,
                    gray_face=gray_face
                )

            emo_text = output_stringify(
                output=pred_emotions,
                tag='emotion',
                top_k=2
            )

            gender_text = output_stringify(
                output=pred_genders,
                tag='gender',
                top_k=2
            )

            age_text = f"age:{str(predict_ages[0]['age'])} con:{str(predict_ages[0]['probability'])}"

            draw_bounding_box(face_coordinates, test_img, color)
            draw_text(face_coordinates, test_img, emo_text,
                      color, -75, -50, 0.5, ui_thickness)
            draw_text(face_coordinates, test_img, gender_text,
                      color, -75, -30, 0.5, ui_thickness)
            draw_text(face_coordinates, test_img, age_text,
                      color, -75, -10, 0.5, ui_thickness)

        resized_img = cv2.resize(test_img, (1024, 768))
        cv2.imshow('Webcam test', resized_img)

        if cv2.waitKey(5) == ord('q') or cv2.waitKey(10) == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_test()
