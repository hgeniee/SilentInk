from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import pickle
import cv2
import numpy as np
import base64

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

model = load_model(r'cnn_model_keras2.h5', compile=False)
with open("hist", "rb") as f:
    hist = pickle.load(f)

image_x, image_y = 50, 50
x, y, w, h = 300, 100, 300, 300

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = np.argmax(pred_probab)
    return pred_class

def preprocess_frame(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[100:100+300, 300:300+300]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            save_img = thresh[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            elif h1 > w1:
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
            return save_img
    return None

def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    roi_thresh = thresh[y:y+h, x:x+w]
    return roi_thresh

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
    lang = get_lang()
    return render_template('eng_to_kor.html', lang=lang)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    encoded_data = data.split(',')[1]
    img_data = base64.b64decode(encoded_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    roi_thresh = get_img_contour_thresh(img)

    result = None
    error = None

    if roi_thresh is not None and roi_thresh.size > 0:
        contour_image = roi_thresh.copy()
        contours, _ = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = roi_thresh[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
                pred_class = keras_predict(save_img)
                result = str(pred_class)
            else:
                error = "Contour too small"
        else:
            error = "No contour found"
    else:
        error = "ROI processing failed"

    thresh_base64 = ""
    if roi_thresh is not None:
        _, buffer = cv2.imencode('.jpg', roi_thresh)
        thresh_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'result': result, 'error': error, 'thresh': thresh_base64})

@app.route('/kor_to_eng')
def kor_to_eng():
    if 'user' not in session:
        return redirect(url_for('login'))
    lang = get_lang()
    return render_template('kor_to_eng.html', lang=lang)

if __name__ == '__main__':
    app.run(debug=True)
