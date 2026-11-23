from functools import lru_cache

import cv2
import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _get_face_cascade():
    cascade_class = getattr(cv2, "CascadeClassifier", None)
    if cascade_class is None:
        return None
    return cascade_class(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def crop_largest_face(image):
    """Return a square crop around the largest detected face, or the full image."""
    face_cascade = _get_face_cascade()
    if face_cascade is None:
        return image

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32),
        )
    except (AttributeError, cv2.error):
        return image
    if len(faces) == 0:
        return image

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    center_x = x + width // 2
    center_y = y + height // 2
    side = int(max(width, height) * 1.35)
    left = max(0, center_x - side // 2)
    top = max(0, center_y - side // 2)
    right = min(rgb.shape[1], left + side)
    bottom = min(rgb.shape[0], top + side)

    return Image.fromarray(rgb[top:bottom, left:right])
