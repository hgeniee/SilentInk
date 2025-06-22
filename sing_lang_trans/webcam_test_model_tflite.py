import os
import time
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.framework.formats import landmark_pb2

from .modules.holistic_module import HolisticDetector  
from .modules.utils import Vector_Normalization        

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUN = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JON = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def compose_jamos(text: str) -> str:
    res, i = [], 0
    while i < len(text):
        if i + 1 >= len(text):
            res.extend(text[i:]); break
        cho, jun = text[i], text[i + 1]
        if cho not in CHO or jun not in JUN:
            res.append(cho); i += 1; continue
        jong = ""
        if i + 2 < len(text) and text[i + 2] in JON[1:]:
            jong = text[i + 2]
        syll = chr(0xAC00 + (CHO.index(cho) * 21 + JUN.index(jun)) * 28 + JON.index(jong))
        res.append(syll)
        i += 3 if jong else 2
    return "".join(res)

actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]
seq_length = 10  

def initialize_detector_and_model() -> Tuple[HolisticDetector, tf.lite.Interpreter]:
    """Mediapipe Detector와 TFLite 모델 로드"""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "multi_hand_gesture_classifier.tflite")

    print("[INFO] 모델 로드:", model_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    detector = HolisticDetector(min_detection_confidence=0.3)
    return detector, interpreter


def process_hand_landmarks(
    right_hand_lmList: landmark_pb2.NormalizedLandmarkList    
) -> np.ndarray:
    """손 랜드마크 → 정규화된 벡터 특징"""
    joint = np.zeros((42, 2))
    for j, lm in enumerate(right_hand_lmList.landmark):
        joint[j] = [lm.x, lm.y]
    vector, angle_label = Vector_Normalization(joint)
    return np.concatenate([vector.flatten(), angle_label.flatten()])


def draw_text_on_image(
    img: np.ndarray,
    text: str,
    pos=(10, 50),
    scale=1.5,
    color=(255, 255, 255),
    thickness=2,
):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img


def predict_action(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    """TFLite 모델 예측 반환"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

def main():
    detector, interpreter = initialize_detector_and_model()
    cap = cv2.VideoCapture(0)

    seq: list[np.ndarray] = []
    action_seq: list[str] = []
    last_action = ""
    collected_jamos = ""
    last_detect_time = time.time()

    display_text = ""        # 화면에 표시할 최종 단어
    display_start_time = 0.0   # 단어 표시 시작 시각

    timeout_seconds = 5        # 손이 사라진 뒤 단어 확정 대기 시간 (3~5)
    show_duration = 3          # 단어 화면 표시 시간

    print("[INFO] 수어 인식 시작 (ESC 키로 종료)")

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)
        now = time.time()

        if right_hand_lmList is not None:
            d = process_hand_landmarks(right_hand_lmList)
            seq.append(d)

            if len(seq) < seq_length:
                last_detect_time = now
                cv2.imshow("img", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # 모델 예측
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = predict_action(interpreter, input_data)
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            # 신뢰도 필터
            if conf < 0.8:
                last_detect_time = now
                cv2.imshow("img", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            action = actions[i_pred]
            action_seq.append(action)

            # 같은 자모가 3프레임 연속일 때만 확정
            if len(action_seq) >= 3 and action_seq[-1] == action_seq[-2] == action_seq[-3]:
                if last_action != action:
                    collected_jamos += action
                    last_action = action
                    print(f"[DEBUG] 자모 확정: {action}")

            last_detect_time = now
            img = draw_text_on_image(img, action, pos=(10, 100), color=(0, 255, 0))

        else:
            if (now - last_detect_time) > timeout_seconds and collected_jamos:
                # 3초 경과 → 단어 확정
                composed_word = compose_jamos(collected_jamos)  
                display_text  = composed_word   
                collected_jamos = ""
                action_seq.clear()
                last_action = ""
                display_start_time = now
                print(f"[INFO] 단어 출력: {display_text}")

        if display_text and (now - display_start_time) < show_duration:
            img = draw_text_on_image(img, display_text, pos=(10, 50), color=(255, 255, 255))
        elif display_text and (now - display_start_time) >= show_duration:
            display_text = "" 

        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
