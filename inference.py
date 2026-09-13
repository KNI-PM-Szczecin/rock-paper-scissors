import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from pathlib import Path
import sys
import os

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return Path(base_path) / relative_path

LANDMARKER_PATH = get_resource_path("hand_landmarker.task")

# tip and PIP indices for each finger (index, middle, ring, pinky)
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def classify(lm) -> tuple[str, int]:
    """Return (gesture, n_extended) using fingertip-vs-PIP rule."""
    extended = [lm[tip].y < lm[pip].y for tip, pip in FINGERS]
    n = sum(extended)

    if n == 0:
        return "ROCK", 0
    if n >= 3:
        return "PAPER", n
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "SCISSORS", 2
    return "?", n


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("Press Q to quit")

cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp_ms += 33
    h, w = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # draw skeleton
        for tip, pip in FINGERS:
            tx, ty = int(lm[tip].x * w), int(lm[tip].y * h)
            px, py = int(lm[pip].x * w), int(lm[pip].y * h)
            color = (0, 230, 0) if lm[tip].y < lm[pip].y else (80, 80, 230)
            cv2.circle(frame, (tx, ty), 8, color, -1)
            cv2.line(frame, (tx, ty), (px, py), (200, 200, 200), 2)

        gesture, n = classify(lm)
        label_color = (0, 230, 0) if gesture != "?" else (0, 140, 230)
        cv2.putText(frame, gesture, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, label_color, 4, cv2.LINE_AA)
    else:
        cv2.putText(frame, "no hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 120, 230), 2, cv2.LINE_AA)

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
