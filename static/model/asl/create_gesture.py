import cv2
import numpy as np
import os
import sqlite3
import random

image_x, image_y = 50, 50

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)"
    try:
        conn.execute(cmd, (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = ? WHERE g_id = ?"
            conn.execute(cmd, (g_name, g_id))
        else:
            print("Doing nothing...")
            return
    conn.commit()

def get_skin_mask(img):
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    skinMask = cv2.inRange(imgYCrCb, lower, upper)
    skinMask = cv2.GaussianBlur(skinMask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return skinMask

def store_images(g_id):
    total_pics = 100
    cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300

    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        skin_mask = get_skin_mask(img)
        roi = skin_mask[y:y+h, x:x+w]
        contours, _ = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = roi[y1:y1+h1, x1:x1+w1]

                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)

                save_img = cv2.resize(save_img, (image_x, image_y))
                
                # 추가 데이터 다양화
                if random.randint(0, 1) == 0:
                    save_img = cv2.flip(save_img, 1)

                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("Skin Mask", skin_mask)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            if not flag_start_capturing:
                frames = 0
        if flag_start_capturing:
            frames += 1
        if pic_no == total_pics:
            break

    cam.release()
    cv2.destroyAllWindows()

# 프로그램 실행 부분
init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)
