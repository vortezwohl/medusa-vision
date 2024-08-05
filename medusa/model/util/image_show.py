import cv2


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 1)


def draw_text(
        coordinates,
        image_array,
        text,
        color,
        x_offset=0,
        y_offset=0,
        font_scale=1,
        thickness=1
):
    x, y = coordinates[:2]
    cv2.putText(
        image_array,
        text,
    (x + x_offset, y + y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )