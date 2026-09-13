import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from pathlib import Path
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return Path(base_path) / relative_path

LANDMARKER_PATH = get_resource_path("hand_landmarker.task")

# tip and PIP indices for each finger (index, middle, ring, pinky)
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


import time
import random

def classify(lm) -> tuple[str, int]:
    """Return (gesture, n_extended) using fingertip-vs-PIP rule."""
    extended = [lm[tip].y < lm[pip].y for tip, pip in FINGERS]
    n = sum(extended)

    if n == 0:
        return "KAMIEN", 0
    if n >= 3:
        return "PAPIER", n
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "NOZYCE", 2
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

# Load custom font
font_path = str(get_resource_path("assets/fonts/game_over.ttf"))
try:
    font_title = ImageFont.truetype(font_path, 120)
    font_gesture = ImageFont.truetype(font_path, 150)
    font_countdown = ImageFont.truetype(font_path, 350)
except IOError:
    font_title = ImageFont.load_default()
    font_gesture = ImageFont.load_default()
    font_countdown = ImageFont.load_default()

timestamp_ms = 0

state = "WAITING"
countdown_start = 0
result_start = 0
computer_gesture = "?"
choices = ["KAMIEN", "PAPIER", "NOZYCE"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    timestamp_ms += 33
    h, w = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    player_gesture = "brak dloni"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        
        # draw skeleton (adjusted for flipped frame, mediapipe handles coordinates as 0-1)
        for tip, pip in FINGERS:
            tx, ty = int(lm[tip].x * w), int(lm[tip].y * h)
            px, py = int(lm[pip].x * w), int(lm[pip].y * h)
            color = (0, 230, 0) if lm[tip].y < lm[pip].y else (80, 80, 230)
            cv2.circle(frame, (tx, ty), 8, color, -1)
            cv2.line(frame, (tx, ty), (px, py), (200, 200, 200), 2)

        player_gesture, n = classify(lm)

    # GAME LOGIC
    if state == "WAITING":
        if player_gesture not in ["brak dloni", "?"]:
            state = "COUNTDOWN"
            countdown_start = time.time()
            computer_gesture = "?"
    elif state == "COUNTDOWN":
        elapsed = time.time() - countdown_start
        if elapsed < 1.0:
            countdown_text = "3"
        elif elapsed < 2.0:
            countdown_text = "2"
        elif elapsed < 3.0:
            countdown_text = "1"
        else:
            state = "RESULT"
            result_start = time.time()
            computer_gesture = random.choice(choices)
            countdown_text = ""
            
    elif state == "RESULT":
        elapsed = time.time() - result_start
        if elapsed > 3.0: # hold result for 3 seconds
            state = "WAITING"
            computer_gesture = "?"

    # DRAW UI WITH PILLOW
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Gracz
    draw.text((20, 20), "Gracz:", font=font_title, fill=(255, 255, 255))
    player_color = (0, 230, 0) if player_gesture not in ["brak dloni", "?"] else (230, 140, 0)
    draw.text((20, 110), player_gesture, font=font_gesture, fill=player_color)

    # Komputer
    bbox_title = draw.textbbox((0, 0), "Komputer:", font=font_title)
    comp_title_w = bbox_title[2] - bbox_title[0]
    draw.text((w - comp_title_w - 20, 20), "Komputer:", font=font_title, fill=(255, 255, 255))
    
    bbox_gest = draw.textbbox((0, 0), computer_gesture, font=font_gesture)
    comp_gest_w = bbox_gest[2] - bbox_gest[0]
    draw.text((w - comp_gest_w - 20, 110), computer_gesture, font=font_gesture, fill=(255, 0, 0))

    # Countdown
    if state == "COUNTDOWN" and countdown_text:
        bbox_cd = draw.textbbox((0, 0), countdown_text, font=font_countdown)
        cd_w = bbox_cd[2] - bbox_cd[0]
        cd_h = bbox_cd[3] - bbox_cd[1]
        text_x = (w - cd_w) // 2
        text_y = (h - cd_h) // 2 - bbox_cd[1]
        draw.text((text_x, text_y), countdown_text, font=font_countdown, fill=(255, 0, 0))

    # Convert back to OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
