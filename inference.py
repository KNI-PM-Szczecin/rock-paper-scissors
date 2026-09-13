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

def draw_text_with_shadow(draw, position, text, font, fill, shadow_color=(0, 0, 0), offset=4):
    """Draws text with a simple drop shadow effect."""
    x, y = position
    draw.text((x + offset, y + offset), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=fill)

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

print("Press Q to quit", flush=True)
print("Press 0-9 to switch cameras", flush=True)

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

# Load and scale gesture icons
gesture_icons = {}
try:
    icon_size = (120, 120)
    rock_img = Image.open(get_resource_path("assets/images/rock.png")).convert("RGBA")
    paper_img = Image.open(get_resource_path("assets/images/paper.png")).convert("RGBA")
    scissors_img = Image.open(get_resource_path("assets/images/sizors.png")).convert("RGBA")
    
    gesture_icons["KAMIEN"] = rock_img.resize(icon_size, Image.Resampling.LANCZOS)
    gesture_icons["PAPIER"] = paper_img.resize(icon_size, Image.Resampling.LANCZOS)
    gesture_icons["NOZYCE"] = scissors_img.resize(icon_size, Image.Resampling.LANCZOS)
except Exception as e:
    print(f"Warning: Could not load gesture icons: {e}", flush=True)

timestamp_ms = 0

state = "WAITING"
countdown_start = 0
result_start = 0
computer_gesture = "?"
choices = ["KAMIEN", "PAPIER", "NOZYCE"]
WINS = {("KAMIEN", "NOZYCE"), ("PAPIER", "KAMIEN"), ("NOZYCE", "PAPIER")}
outcome_text = ""
outcome_color = (255, 255, 255)

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

    countdown_text = ""

    # GAME LOGIC
    if state == "WAITING":
        if player_gesture not in ["brak dloni", "?"]:
            state = "COUNTDOWN"
            countdown_start = time.time()
            computer_gesture = "?"
            outcome_text = ""
            
    if state == "COUNTDOWN":
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
            
            if player_gesture == computer_gesture:
                outcome_text = "REMIS"
                outcome_color = (255, 255, 0) # Yellow
            elif (player_gesture, computer_gesture) in WINS:
                outcome_text = "WYGRALES!"
                outcome_color = (0, 255, 0) # Green
            else:
                outcome_text = "PRZEGRALES"
                outcome_color = (255, 0, 0) # Red
            
    elif state == "RESULT":
        elapsed = time.time() - result_start
        if elapsed > 3.0: # hold result for 3 seconds
            state = "WAITING"
            computer_gesture = "?"
            outcome_text = ""

    # DRAW UI WITH PILLOW
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Gracz
    draw_text_with_shadow(draw, (20, 20), "Gracz:", font=font_title, fill=(255, 255, 255))
    player_color = (0, 230, 0) if player_gesture not in ["brak dloni", "?"] else (230, 140, 0)
    draw_text_with_shadow(draw, (20, 110), player_gesture, font=font_gesture, fill=player_color)
    if player_gesture in gesture_icons:
        frame_pil.paste(gesture_icons[player_gesture], (20, 230), mask=gesture_icons[player_gesture])

    # Komputer
    bbox_title = draw.textbbox((0, 0), "Komputer:", font=font_title)
    comp_title_w = bbox_title[2] - bbox_title[0]
    draw_text_with_shadow(draw, (w - comp_title_w - 20, 20), "Komputer:", font=font_title, fill=(255, 255, 255))
    
    bbox_gest = draw.textbbox((0, 0), computer_gesture, font=font_gesture)
    comp_gest_w = bbox_gest[2] - bbox_gest[0]
    draw_text_with_shadow(draw, (w - comp_gest_w - 20, 110), computer_gesture, font=font_gesture, fill=(255, 0, 0))
    if computer_gesture in gesture_icons:
        icon_w, _ = gesture_icons[computer_gesture].size
        frame_pil.paste(gesture_icons[computer_gesture], (w - icon_w - 20, 230), mask=gesture_icons[computer_gesture])

    # Countdown
    if state == "COUNTDOWN" and countdown_text:
        bbox_cd = draw.textbbox((0, 0), countdown_text, font=font_countdown)
        cd_w = bbox_cd[2] - bbox_cd[0]
        cd_h = bbox_cd[3] - bbox_cd[1]
        text_x = (w - cd_w) // 2
        text_y = (h - cd_h) // 2 - bbox_cd[1]
        draw_text_with_shadow(draw, (text_x, text_y), countdown_text, font=font_countdown, fill=(255, 0, 0))

    # Outcome
    if outcome_text:
        bbox_out = draw.textbbox((0, 0), outcome_text, font=font_gesture)
        out_w = bbox_out[2] - bbox_out[0]
        out_h = bbox_out[3] - bbox_out[1]
        out_x = (w - out_w) // 2
        out_y = (h * 3) // 4 - out_h // 2 - bbox_out[1]
        draw_text_with_shadow(draw, (out_x, out_y), outcome_text, font=font_gesture, fill=outcome_color)

    # Convert back to OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif ord('0') <= key <= ord('9'):
        new_cam_idx = key - ord('0')
        
        # On Windows, using cv2.CAP_DSHOW can prevent hanging when querying invalid cameras.
        # But for cross-platform simplicity and safety, we try normal capture first.
        # In a real app we might use system APIs, but OpenCV handles invalid indices 
        # by returning False from isOpened() or read()
        new_cap = cv2.VideoCapture(new_cam_idx)
        
        if new_cap.isOpened():
            # Check if it actually produces a frame
            test_ret, _ = new_cap.read()
            if test_ret:
                cap.release()
                cap = new_cap
                print(f"Switched to camera {new_cam_idx}", flush=True)
            else:
                new_cap.release()
                print(f"Camera {new_cam_idx} opened but cannot read frames.", flush=True)
        else:
            new_cap.release()
            print(f"Camera {new_cam_idx} is not available.", flush=True)

cap.release()
cv2.destroyAllWindows()
landmarker.close()
