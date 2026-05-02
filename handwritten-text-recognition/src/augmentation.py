import cv2
import numpy as np
import random

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def add_noise(image):
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    return cv2.add(image, noise)

def scale_image(image, scale=1.5):
    return cv2.resize(image, None, fx=scale, fy=scale)

def augment_image(image, method):
    if method == "rotate":
        angle = random.choice([-10, -5, 5, 10])
        return rotate_image(image, angle)

    elif method == "noise":
        return add_noise(image)

    elif method == "scale":
        return scale_image(image, 1.5)

    elif method == "rotate_noise":
        img = rotate_image(image, random.choice([-10, 10]))
        return add_noise(img)

    else:
        return image