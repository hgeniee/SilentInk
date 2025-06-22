# SilentInk

> 영어권 농인(ASL)과 한국 농인(KSL) 간 실시간 수어 통역 서비스

---

## 📖 프로젝트 소개
`SilentInk`는 서로 다른 수어 체계를 사용하는 사용자(영어권 농인과 한국 농인) 간의 **실시간 커뮤니케이션**을 지원합니다.  
카메라에 비친 수어 동작을 텍스트로 변환하고, 자동 번역하여 상대방 화면에 출력하는 “수어 → 텍스트 → 번역 → 텍스트 & 음성출력” 파이프라인을 구현했습니다.

---

## 🎯 주요 기능
1. **수어 입력(영상)**  
   - 웹캠(또는 내장 카메라)으로 ASL/KSL 동작을 실시간 캡처

2. **수어 인식 → 텍스트 변환**  
   - ASL: CNN 기반 딥러닝 모델  
   - KSL: KNN(최근접 이웃) 알고리즘

3. **텍스트 자동 번역**  
   - Google Translate API 연동  
   - ASL 인식 결과(영어) → 한글  
   - KSL 인식 결과(한국어) → 영어

4. **음성 출력**  
   - 브라우저 내장 TTS(Text-to-Speech) 기능인 window.speechSynthesis를 사용하여 음성 출력

---

## 🖥️ 프로젝트 설치 및 실행 방법
**필수 패키지 설치**
```bash
pip install -r requirements.txt
```

**프로그램 실행**
```bash
python app.py
```
실행 후 웹 브라우저에서 회원가입 또는 로그인 (영어/한국어 선택 가능) →
메인 페이지에서 "영어 수화 ↔ 한국어 수화 번역" 중 원하는 모드를 선택 →
웹캠을 통해 실시간으로 수화를 인식하여 번역 결과를 제공합니다.

## **수화를 학습시키고 싶다면**

### 🏷️ eng → kor
1. `create_gestures.py` 실행  
2. `load_images.py` 실행  
3. `cnn_model_train.py` 실행  
4. `final.py` 실행해서 확인  
--
### 🏷️ kor → eng
1. `create_dataset_from_video.py` 실행 (데이터 전처리)
2. Jupyter 노트북 `sing_lang_trans/train_hand_gesture.ipynb` 전체 실행  
3. `webcam_test_model_tflite.py` 실행해 실시간 확인  

> ✅ LSTM 기반 모델로 수어 동작 시퀀스를 학습  
> ✅ 학습된 TFLite 모델을 사용해 실시간 수어 → 영어 자막 출력  

#### 🔧 모델 구성 (train_hand_gesture.ipynb)

- LSTM → Dropout → Dense → Dropout → Dense  
- L2 Regularization (0.01)  
- ReLU 활성화 함수  
- Dropout(0.3) *2  
- CategoricalCrossEntropy 손실 함수  
- Adam 옵티마이저  
- ReduceLROnPlateau (factor 0.5, patience 50)  
- EarlyStopping (patience 20, 최대 200 epoch)

---

## 🛠️ 기술 스택 & 도구
영상 처리: OpenCV, MediaPipe <br>
ASL 인식: TensorFlow / PyTorch 기반 CNN <br>
KSL 인식: scikit-learn KNN <br>
번역 API: Google Translate REST API <br>
Backend: Flask <br>
Frontend: HTML, CSS, JavaScript <br>
버전 관리: Git & GitHub <br>
개발 환경: 로컬 PC + GPU 서버(학습 시) <br>

## 📂 프로젝트 파일 구조
```plaintext
SILENTINK/
├── .github/                   # GitHub workflow & actions (if applicable)
├── gestures/                  # 수어 제스처 관련 모듈
│
├── static/                    # 정적 파일 (CSS, 이미지 등)
│   ├── design/                # 스타일시트
│   │   ├── eng_to_kor.css
│   │   ├── login.css
│   │   ├── main.css
│   │   └── signup.css
│   ├── images/                # 이미지 리소스
│   │   └── silent_logo.png
│   └── model/asl/             # ASL 수어 인식 모델 관련 코드
│       ├── cnn_model_train.py
│       ├── create_gesture.py
│       ├── display_gestures.py
│       ├── final.py
│       ├── hist/
│       ├── load_images.py
│       └── set_hand_histogram.py
│
├── templates/                 # HTML 템플릿 (Flask 연동용)
│   ├── eng_to_kor.html
│   ├── kor_to_eng.html
│   ├── login.html
│   ├── main.html
│   └── signup.html
│
├── app.py                     # Flask 서버 실행 파일
├── cnn_model_keras2.h5        # 학습된 CNN 모델 파일
├── full_img.jpg               # 예제 이미지
├── gesture_db.db              # 수어 DB (KNN 기반)
├── hist/                      # 손 히스토그램 데이터
├── README.md                  # 프로젝트 설명 문서
├── speech.mp3                 # 번역된 텍스트 음성 출력
├── test_images/               # 테스트 이미지셋
├── test_labels/
├── train_images/              # 학습 이미지셋
├── train_labels/
├── val_images/                # 검증 이미지셋
└── val_labels/
```

---

## 👥 팀원
숙명여자대학교 수학과 20 최윤녕<br>
숙명여자대학교 인공지능공학부 23 이현진<br>
숙명여자대학교 인공지능공학부 23 황유림

---

## 🔗 원본 프로젝트 출처

본 프로젝트는 다음 오픈소스 프로젝트를 기반으로 개발되었습니다:

- **Sign Language Interpreter using Deep Learning**  
  - GitHub 링크: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning.git
    
- **hearing_impaired_helper_make_model**  
  - GitHub 링크: https://github.com/Ghoney99/hearing_impaired_helper_make_model.git
 
---

## 📜 라이센스 
MIT License

Copyright (c) 2025 Yoonryung Choi, Hyunchin Lee, Yurim Hwang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


---
