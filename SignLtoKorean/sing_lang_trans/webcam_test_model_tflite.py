import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
import time

# -------------------------------------------------
# 1) 제스처(자모) 목록 정의
# -------------------------------------------------
# 에러 “name 'actions' is not defined”를 방지하려면,
# 맨 위에 actions 리스트를 반드시 선언해야 합니다.
actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]

# -------------------------------------------------
# 2) 시퀀스 길이 정의
# -------------------------------------------------
# TFLite 모델이 연속으로 받아들이는 프레임 수
seq_length = 10

# -------------------------------------------------
# 3) 한글 초성·중성·종성 테이블
# -------------------------------------------------
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUN = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JON = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ",
       "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
       "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ",
       "ㅌ", "ㅍ", "ㅎ"]

# -------------------------------------------------
# 4) 자모 세 글자를 조합하여 한 음절로 만드는 함수
# -------------------------------------------------
def join_jamos_manual(jamos):
    """
    jamos: [초성, 중성, (종성)] 형태로 들어옴.
    종성이 없으면 jamos 길이가 2, 있으면 3.
    """
    cho = jamos[0]
    jung = jamos[1]
    jong = jamos[2] if len(jamos) == 3 else ""
    try:
        cho_idx = CHO.index(cho)
        jung_idx = JUN.index(jung)
        jong_idx = JON.index(jong) if jong else 0
        code_point = 0xAC00 + (cho_idx * 21 + jung_idx) * 28 + jong_idx
        return chr(code_point)
    except ValueError:
        # 예외 발생 시, 입력된 자모를 그대로 합쳐서 반환
        return "".join(jamos)

# -------------------------------------------------
# 5) 자모 리스트를 순회하면서 음절 단위로 묶는 함수 (개선된 버전)
# -------------------------------------------------
def compose_all_jamo(jamos):
    """
    jamos: ['ㅇ','ㅜ','ㅇ','ㅠ'] 등 자모 시퀀스.
    초성+중성(+종성) 단위로 묶어서 문자열을 반환.
    개선: 종성이 다음 음절의 초성('ㅇ')일 때 중복 조합을 방지.
    """
    result = ""
    i = 0
    n = len(jamos)

    while i < n:
        # 초성이고, 다음에 중성이 와야 음절 조합 시도
        if jamos[i] in CHO and i + 1 < n and jamos[i + 1] in JUN:
            cho = jamos[i]
            jung = jamos[i + 1]
            jong = ""
            # 종성 후보가 있으면 검사
            if i + 2 < n and jamos[i + 2] in JON[1:]:
                # 만약 i+3 자리에 중성이 온다면,
                # jamos[i+2]는 종성이 아니라 다음 음절의 초성일 가능성이 높음
                if not (i + 3 < n and jamos[i + 3] in JUN):
                    jong = jamos[i + 2]
                    i += 3
                else:
                    # 다음에 중성이 오므로 종성으로 취급하지 않음
                    i += 2
            else:
                i += 2
            result += join_jamos_manual([cho, jung, jong] if jong else [cho, jung])
        else:
            # 음절 조합이 불가능한 단독 자모(자음 또는 모음)인 경우
            result += jamos[i]
            i += 1

    return result

# -------------------------------------------------
# 6) 전역 변수
# -------------------------------------------------
jamo_sequence = []       # 현재까지 인식된 자모들을 순차적으로 보관
last_input_time = time.time()

fontpath = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
try:
    font = ImageFont.truetype(fontpath, 40)
except OSError:
    print("[경고] 폰트를 불러올 수 없습니다. 기본 폰트를 사용합니다.")
    font = ImageFont.load_default()

# -------------------------------------------------
# 7) Mediapipe + TFLite 모델 초기화 함수
# -------------------------------------------------
def initialize_detector_and_model():
    detector = hm.HolisticDetector(min_detection_confidence=0.3)
    interpreter = tf.lite.Interpreter(
        model_path="/Users/choiyoonryung/Desktop/녕이네/대학생녕/it녕/"
                   "오픈소스프로그래밍_프로젝트/SignLtoKorean/"
                   "models/multi_hand_gesture_classifier.tflite"
    )
    interpreter.allocate_tensors()
    return detector, interpreter

# -------------------------------------------------
# 8) 손 랜드마크를 벡터+각도로 변환하는 함수
# -------------------------------------------------
def process_hand_landmarks(right_hand_lmList):
    joint = np.zeros((42, 2))
    for j, lm in enumerate(right_hand_lmList.landmark):
        joint[j] = [lm.x, lm.y]
    vector, angle_label = Vector_Normalization(joint)
    return np.concatenate([vector.flatten(), angle_label.flatten()])

# -------------------------------------------------
# 9) TFLite 모델로 제스처 예측하는 함수
# -------------------------------------------------
def predict_action(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    return y_pred[0]

# -------------------------------------------------
# 10) 이미지 위에 텍스트(자막)를 그려주는 함수
# -------------------------------------------------
def draw_text_on_image(img, text):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font_size = 100
    try:
        font = ImageFont.truetype(fontpath, font_size)
    except OSError:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
    except AttributeError:
        text_width, _ = font.getsize(text)
    img_width, _ = img_pil.size
    x = (img_width - text_width) // 2
    draw.text((x, 30), text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

# -------------------------------------------------
# 11) 메인 루프
# -------------------------------------------------
def main():
    global last_input_time, jamo_sequence

    detector, interpreter = initialize_detector_and_model()
    cap = cv2.VideoCapture(1)

    seq = []
    action_seq = []
    last_action = None

    has_displayed_final = False
    displayed_text = ""

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        current_time = time.time()

        # ---------------------------------------------
        # (1) 5초 동안 신규 입력이 없으면 최종 단어 표시
        # ---------------------------------------------
        if not has_displayed_final and jamo_sequence and (current_time - last_input_time > 5):
            displayed_text = compose_all_jamo(jamo_sequence)
            has_displayed_final = True
            jamo_sequence = []

        # 새로운 손 랜드마크가 감지되면 최종 표시 초기화
        if right_hand_lmList is not None:
            if has_displayed_final:
                has_displayed_final = False
                displayed_text = ""
                jamo_sequence = []
                seq = []
                action_seq = []
                last_action = None

            # 손 랜드마크 → 자모 예측 로직
            d = process_hand_landmarks(right_hand_lmList)
            seq.append(d)
            if len(seq) < seq_length:
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = predict_action(interpreter, input_data)

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            if conf < 0.7:
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            this_action = '?'
            # 같은 제스처가 3번 연속으로 나오면 확정
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                if last_action != this_action:
                    last_action = this_action
                    jamo_sequence.append(this_action)
                    last_input_time = current_time

            # 자모 시퀀스 길이에 따라 화면에 텍스트 출력
            if len(jamo_sequence) >= 2:
                try:
                    syllable = join_jamos_manual(jamo_sequence[-3:])
                    img = draw_text_on_image(img, syllable)
                except:
                    img = draw_text_on_image(img, this_action)
            else:
                img = draw_text_on_image(img, this_action)

        # 완성된 단어를 고정해서 화면에 띄우기
        elif has_displayed_final:
            img = draw_text_on_image(img, displayed_text)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
