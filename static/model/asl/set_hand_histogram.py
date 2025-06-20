import cv2
import numpy as np

def get_skin_mask(img):
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    skinMask = cv2.inRange(imgYCrCb, lower, upper)
    skinMask = cv2.GaussianBlur(skinMask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return skinMask

def run_skin_detection():
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        skin_mask = get_skin_mask(img)
        skin_mask_bgr = cv2.merge([skin_mask]*3)

        cv2.imshow("Original", img)
        cv2.imshow("Skin Mask", skin_mask_bgr)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

run_skin_detection()