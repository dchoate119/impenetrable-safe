import base64
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App, Bridge

CODE = [5, 4, 1]

CLOSED_ANGLE = 170
OPEN_ANGLE = 90

HOLD_TIME = 1.0      # seconds a digit must be held to register
CONFIDENCE = 0.6     # min smoothed probability to count a digit or fist
WRONG_LOCKOUT = 3.0  # seconds of no input after a wrong code
OPEN_TIMEOUT = 30.0  # seconds before the safe relocks by itself

BOX = 300        # must match collect.py --box
SAVE_SIZE = 128  # must match collect.py --size
IMG = 64         # must match the notebook's IMG
SMOOTH = 8       # average predictions over this many frames
SEND_EVERY = 2   # send every Nth frame to the WebUI
MODEL_PATH = Path(__file__).parent / "model.onnx"

ui = WebUI()
camera = Camera(resolution=(640, 480), fps=30)
net = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
history = deque(maxlen=SMOOTH)
angle = CLOSED_ANGLE
frame_count = 0
last_time = time.monotonic()
fps = 0.0

# Code entry state
status = "locked"   # "locked" | "open" | "wrong"
entered = []        # digits registered so far
armed = True        # False after a digit registers, until a fist is shown
hold_digit = None   # digit currently being held
hold_start = 0.0
status_since = 0.0  # when "open" or "wrong" started


# --- Servo and lock state ----------------------------------------------------
def move(new_angle):
    global angle
    angle = new_angle
    print(f"servo {angle}", flush=True)
    Bridge.call("servo", angle)
    ui.send_message("angle", {"angle": angle})


def reset_entry():
    global entered, armed, hold_digit
    entered, armed, hold_digit = [], True, None


def set_status(new_status):
    global status, status_since
    status, status_since = new_status, time.monotonic()
    print(f"status {status}", flush=True)


def open_safe():
    reset_entry()
    set_status("open")
    move(OPEN_ANGLE)


def close_safe():
    reset_entry()
    set_status("locked")
    move(CLOSED_ANGLE)


ui.on_message("open", lambda sid, data: open_safe())
ui.on_message("reset", lambda sid, data: close_safe())
ui.on_connect(lambda sid: ui.send_message("angle", {"angle": angle}, room=sid))


# --- Model (same steps as laptop/live_test.py) -------------------------------
def preprocess(frame):
    h, w = frame.shape[:2]
    x0, y0 = (w - BOX) // 2, (h - BOX) // 2
    gray = cv2.cvtColor(frame[y0:y0 + BOX, x0:x0 + BOX], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)[None, None], (x0, y0)  # (1, 1, IMG, IMG)


def predict(x):
    scores = net.run(None, {"image": x})[0][0]
    e = np.exp(scores - scores.max())
    return e / e.sum()  # softmax


# --- Code entry: hold a digit 1 s, fist between digits -----------------------
def update_entry(probs, now):
    """Returns hold progress 0-1 for the UI."""
    global armed, hold_digit, hold_start

    if status == "open":
        if now - status_since > OPEN_TIMEOUT:
            close_safe()
        return 0.0
    if status == "wrong":
        if now - status_since > WRONG_LOCKOUT:
            reset_entry()
            set_status("locked")
        return 0.0

    pred, conf = int(probs.argmax()), float(probs.max())
    if conf < CONFIDENCE:
        hold_digit = None
        return 0.0
    if pred == 0:
        armed, hold_digit = True, None
        return 0.0
    if not armed:
        return 0.0

    if pred != hold_digit:
        hold_digit, hold_start = pred, now
    progress = (now - hold_start) / HOLD_TIME
    if progress < 1.0:
        return progress

    entered.append(pred)
    armed, hold_digit = False, None
    print(f"digit {pred}  entered {entered}", flush=True)
    if len(entered) == len(CODE):
        if entered == CODE:
            open_safe()
        else:
            set_status("wrong")
    return 1.0


# --- WebUI frame: mirrored, with the hand box ---------------------------------
def send_frame(frame, origin, probs, progress):
    shown = cv2.flip(frame, 1)
    w = frame.shape[1]
    x0, y0 = origin
    cv2.rectangle(shown, (w - x0 - BOX, y0), (w - x0, y0 + BOX), (0, 255, 0), 2)
    ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        ui.send_message("frame", {
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
            "probs": [round(float(p), 3) for p in probs],
            "fps": round(fps, 1),
            "status": status,
            "entered": len(entered),
            "code_len": len(CODE),
            "armed": armed,
            "progress": round(min(progress, 1.0), 2),
        })


# --- Main loop ---------------------------------------------------------------
def loop():
    global frame_count, last_time, fps
    frame = camera.capture()
    if frame is None:
        return

    x, origin = preprocess(frame)
    history.append(predict(x))
    probs = np.mean(history, axis=0)

    now = time.monotonic()
    progress = update_entry(probs, now)

    frame_count += 1
    fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
    last_time = now

    if frame_count % SEND_EVERY == 0:
        send_frame(frame, origin, probs, progress)
    if frame_count % 30 == 0:
        print(f"pred {int(probs.argmax())} ({probs.max():.0%})  fps {fps:.1f}", flush=True)


camera.start()
close_safe()
try:
    App.run(user_loop=loop)
finally:
    camera.stop()
    Bridge.call("servo", CLOSED_ANGLE)
