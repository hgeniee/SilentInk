from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

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
            session['lang'] = language  # 세션에 최종 선택된 언어 저장
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

        print("디버깅 - user 데이터:", user)

        if user and check_password_hash(user['password'], password):
            print("디버깅 - user['language'] 값:", user['language'])

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
    print("디버깅 - 현재 main()에서 읽은 lang:", lang)

    return render_template('main.html', username=session['user'], lang=lang)

@app.route('/eng_to_kor')
def eng_to_kor():
    if 'user' not in session:
        return redirect(url_for('login'))
    lang = get_lang()
    return render_template('eng_to_kor.html', lang=lang)

@app.route('/kor_to_eng')
def kor_to_eng():
    if 'user' not in session:
        return redirect(url_for('login'))
    lang = get_lang()
    return render_template('kor_to_eng.html', lang=lang)

if __name__ == '__main__':
    app.run(debug=True)
