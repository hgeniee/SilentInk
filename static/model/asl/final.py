import cv2
import numpy as np
from keras.models import load_model

model = load_model("cnn_model_keras2.h5")
image_x, image_y = 50, 50

def keras_predict(img):
    img = cv2.resize(img, (image_x, image_y)).astype(np.float32) / 255.0
    img = img.reshape(1, image_x, image_y, 1)
    pred = model.predict(img)[0]
    return np.argmax(pred), max(pred)

def get_skin_mask(img):
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(imgYCrCb, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

cam = cv2.VideoCapture(0)
x, y, w, h = 300, 100, 300, 300

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    mask = get_skin_mask(img)
    roi = mask[y:y+h, x:x+w]
    contours, _ = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            hand_img = roi[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                hand_img = cv2.copyMakeBorder(hand_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
            elif h1 > w1:
                hand_img = cv2.copyMakeBorder(hand_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)
            pred_class, confidence = keras_predict(hand_img)
            cv2.putText(img, f"Class: {pred_class} ({confidence*100:.2f}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Recognition", img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
