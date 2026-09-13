from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import threading
import random
import time
from pathlib import Path
import sys
import os

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return Path(base_path) / relative_path

app = Flask(__name__, template_folder=str(get_resource_path('templates')))

LANDMARKER_PATH = get_resource_path("hand_landmarker.task")
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]
CHOICES = ["rock", "paper", "scissors"]
WINS = {("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")}

scores = {"player": 0, "computer": 0, "draws": 0}
output_frame = None
current_gesture = None
state_lock = threading.Lock()


def classify(lm):
    extended = [lm[tip].y < lm[pip].y for tip, pip in FINGERS]
    n = sum(extended)
    if n == 0:
        return "rock"
    if n >= 3:
        return "paper"
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "scissors"
    return None


def camera_loop():
    global output_frame, current_gesture

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        gesture = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            gesture = classify(lm)
            for tip, pip in FINGERS:
                tx, ty = int(lm[tip].x * w), int(lm[tip].y * h)
                color = (0, 230, 0) if lm[tip].y < lm[pip].y else (60, 60, 200)
                cv2.circle(frame, (tx, ty), 8, color, -1)

        if gesture:
            cv2.putText(frame, gesture.upper(), (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 230, 0), 3, cv2.LINE_AA)
        elif result.hand_landmarks:
            cv2.putText(frame, "?", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 80, 230), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "brak reki", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 2, cv2.LINE_AA)

        with state_lock:
            output_frame = frame.copy()
            current_gesture = gesture


threading.Thread(target=camera_loop, daemon=True).start()


def generate_frames():
    while True:
        with state_lock:
            frame = output_frame.copy() if output_frame is not None else None

        if frame is None:
            time.sleep(0.03)
            continue

        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + encoded.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/play', methods=['POST'])
def play():
    with state_lock:
        player = current_gesture

    if player is None:
        return jsonify({"error": "no_gesture"}), 400

    computer = random.choice(CHOICES)

    if player == computer:
        outcome = "draw"
        scores["draws"] += 1
    elif (player, computer) in WINS:
        outcome = "win"
        scores["player"] += 1
    else:
        outcome = "lose"
        scores["computer"] += 1

    return jsonify({"player": player, "computer": computer,
                    "outcome": outcome, "scores": dict(scores)})


@app.route('/api/reset', methods=['POST'])
def reset():
    scores.update({"player": 0, "computer": 0, "draws": 0})
    return jsonify({"ok": True})


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
