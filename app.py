from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 세션을 위한 비밀 키

# 임시 유저 저장소 (나중에 DB로 대체 가능)
users = {}

# 기본 홈 → 로그인 페이지로 리디렉션
@app.route('/')
def home():
    return redirect(url_for('login'))

# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.get(email)

        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']
            flash('로그인 성공', 'success')
            return redirect(url_for('main'))
        else:
            flash('이메일 또는 비밀번호 오류입니다.', 'error')

    return render_template('login.html')

# 회원가입 페이지
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if email in users:
            flash('이미 등록된 이메일입니다.', 'error')
        else:
            users[email] = {
                'name': name,
                'password': generate_password_hash(password)
            }
            flash('회원가입 성공! 로그인 해주세요.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')

# 메인 페이지 (번역 방향 선택)
@app.route('/main')
def main():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('main.html', username=session['user'])

# 영어 → 한국어 번역 페이지
@app.route('/eng_to_kor', methods=['GET', 'POST'])
def eng_to_kor():
    if 'user' not in session:
        return redirect(url_for('login'))

    translated = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        # 여기에 모델 호출 코드 넣을 수 있음
        translated = f"(번역결과) {input_text}"  # 임시 출력

    return render_template('eng_to_kor.html', translated=translated)

# 한국어 → 영어 번역 페이지
@app.route('/kor_to_eng', methods=['GET', 'POST'])
def kor_to_eng():
    if 'user' not in session:
        return redirect(url_for('login'))

    translated = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        translated = f"(Translated) {input_text}"  # 임시 출력

    return render_template('kor_to_eng.html', translated=translated)

# 로그아웃
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('로그아웃 되었습니다.', 'success')
    return redirect(url_for('login'))

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True)
