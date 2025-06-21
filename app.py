from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import pickle
import cv2
import numpy as np
import base64
import sqlite3
import time

from sing_lang_trans.webcam_test_model_tflite import (
    initialize_detector_and_model,
    process_hand_landmarks,
    predict_action,
    actions,
    seq_length
)

# 서버 구동 시 한 번만 모델 로딩
detector, interpreter = initialize_detector_and_model()

# 순차 예측을 위한 전역 변수
seq = []
action_seq = []
last_action = None

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.jinja_env.cache = {}

users = {}

messages = {
    'signup_success': {
        'ko': '회원가입 성공! 로그인 해주세요.',
        'en': 'Sign up successful! Please log in.'
    },
    'signup_duplicate': {
        'ko': '이미 등록된 이메일입니다.',
        'en': 'Email is already registered.'
    },
    'login_success': {
        'ko': '로그인 성공',
        'en': 'Login successful'
    },
    'login_fail': {
        'ko': '이메일 또는 비밀번호가 잘못되었습니다.',
        'en': 'Incorrect email or password.'
    }
}

def load_label_map():
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.execute("SELECT g_id, g_name FROM gesture")
    label_map = {row[0]: row[1] for row in cursor}
    conn.close()
    return label_map

model = load_model('cnn_model_keras2.h5', compile=False)
label_map = load_label_map()
image_x, image_y = 50, 50
x, y, w, h = 300, 100, 300, 300

def keras_predict(img):
    img = cv2.resize(img, (image_x, image_y)).astype(np.float32) / 255.0
    img = img.reshape(1, image_x, image_y, 1)
    pred = model.predict(img)[0]
    pred_class = np.argmax(pred)
    return pred_class

def get_skin_mask(img):
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(imgYCrCb, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def get_lang():
    return session.get('lang', 'ko')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    lang = get_lang()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        language = request.form['language']
        if email in users:
            flash(messages['signup_duplicate'][lang], 'error')
        else:
            users[email] = {
                'name': name,
                'password': generate_password_hash(password),
                'language': language
            }
            session['lang'] = language
            flash(messages['signup_success'][language], 'success')
            return redirect(url_for('login'))
    return render_template('signup.html', lang=lang)

@app.route('/login', methods=['GET', 'POST'])
def login():
    lang = get_lang()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.get(email)
        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']
            if 'lang' not in session:
                session['lang'] = user['language']
            lang = session['lang']
            flash(messages['login_success'][lang], 'success')
            return redirect(url_for('main'))
        else:
            flash(messages['login_fail'][lang], 'error')
    return render_template('login.html', lang=lang)

@app.route('/main')
def main():
    if 'user' not in session:
        return redirect(url_for('login'))
    lang = get_lang()
    return render_template('main.html', username=session['user'], lang=lang)

@app.route('/eng_to_kor')
def eng_to_kor():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('eng_to_kor.html', lang=session.get('lang', 'ko'))

collected_jamos = ''
last_jamo_time  = time.time()
TIMEOUT_SECONDS = 3   

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUN = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JON = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def compose_jamos(text: str) -> str:
    res, i = [], 0
    while i < len(text):
        if i + 1 >= len(text):
            res.extend(text[i:]); break
        cho, jun = text[i], text[i+1]
        if cho not in CHO or jun not in JUN:
            res.append(cho); i += 1; continue
        jong = ""
        if i + 2 < len(text) and text[i+2] in JON[1:]:
            jong = text[i+2]
        syll = chr(0xAC00 + (CHO.index(cho)*21 + JUN.index(jun))*28 + JON.index(jong))
        res.append(syll)
        i += 3 if jong else 2
    return "".join(res)

@app.route('/predict', methods=['POST'])
def predict():
    global seq, action_seq, last_action
    global collected_jamos, last_jamo_time

    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img = detector.findHolistic(frame, draw=True)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    result = None
    result_type = 'char'  

    if right_hand_lmList:
        feat = process_hand_landmarks(right_hand_lmList)
        seq.append(feat)

        if len(seq) >= seq_length:
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = predict_action(interpreter, input_data)
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf > 0.9:
                action = actions[i_pred]
                action_seq.append(action)

                if (
                    len(action_seq) >= 3 and
                    action_seq[-3:] == [action] * 3 and
                    last_action != action
                ):
                    last_action = action
                    collected_jamos += action
                    last_jamo_time = time.time()
                    result = action
                    result_type = 'char'

    if collected_jamos and (time.time() - last_jamo_time > TIMEOUT_SECONDS):
        result = compose_jamos(collected_jamos)
        result_type = 'word'
        collected_jamos = ''
        action_seq.clear()
        last_action = None

    _, buf = cv2.imencode('.jpg', img)
    frame_b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        'result': result,
        'type': result_type,
        'frame': frame_b64
    })

@app.route('/predict-asl', methods=['POST'])
def predict_asl():
    data = request.json['image']
    encoded_data = data.split(',')[1]
    img_data = base64.b64decode(encoded_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.flip(img, 1)

    skin_mask = get_skin_mask(img)
    roi = skin_mask[y:y+h, x:x+w]
    contours, _ = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result = None
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            hand_img = roi[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                hand_img = cv2.copyMakeBorder(hand_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
            elif h1 > w1:
                hand_img = cv2.copyMakeBorder(hand_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, 0)

            pred_class = keras_predict(hand_img)  
            result = label_map.get(pred_class, "Unknown")

    _, buffer = cv2.imencode('.jpg', roi)
    thresh_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'result': result, 'thresh': thresh_base64})


@app.route('/kor_to_eng')
def kor_to_eng():
    if 'user' not in session:
        return redirect(url_for('login'))
    lang = get_lang()
    return render_template('kor_to_eng.html', lang=lang)

if __name__ == '__main__':
    app.run(debug=True)
