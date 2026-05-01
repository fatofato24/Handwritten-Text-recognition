import cv2

def preprocess_image(image, method="gray"):
    
    if method == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif method == "resize":
        return cv2.resize(image, None, fx=2, fy=2)

    elif method == "gray_resize":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=2, fy=2)

    elif method == "blur":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    elif method == "threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        return thresh

    elif method == "adaptive":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    elif method == "blur_threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
        return thresh

    else:
        return image