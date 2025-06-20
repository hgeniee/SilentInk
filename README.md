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
