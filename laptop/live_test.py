"""Live finger-count detector using the trained model.pt.

    python laptop/live_test.py

Hold your hand in the box. q = quit.
"""
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

CAM = 5          # SK-C201 USB webcam
BOX = 300        # must match collect.py --box
SAVE_SIZE = 128  # must match collect.py --size
SMOOTH = 8       # average predictions over this many frames
MODEL_PATH = Path(__file__).parent / "model.pt"


# --- 1. Model (must match the notebook's FingerNet) -------------------------
def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class FingerNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


# --- 2. Preprocess: same steps as collect.py + notebook loading -------------
def preprocess(frame, img):
    h, w = frame.shape[:2]
    x0, y0 = (w - BOX) // 2, (h - BOX) // 2
    gray = cv2.cvtColor(frame[y0:y0 + BOX, x0:x0 + BOX], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.resize(gray, (img, img), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]  # (1, 1, img, img)
    return x, gray, (x0, y0)


# --- 3. Overlay -------------------------------------------------------------
def draw(frame, origin, model_view, probs):
    shown = cv2.flip(frame, 1)
    w = frame.shape[1]
    x0, y0 = origin
    cv2.rectangle(shown, (w - x0 - BOX, y0), (w - x0, y0 + BOX), (0, 255, 0), 2)

    pred = int(probs.argmax())
    label = "fist/none" if pred == 0 else str(pred)
    cv2.putText(shown, f"{label}  {probs[pred]:.0%}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Probability bars for classes 0-5
    for c, p in enumerate(probs):
        y = 70 + c * 25
        cv2.rectangle(shown, (40, y), (40 + int(p * 150), y + 18),
                      (0, 255, 0) if c == pred else (180, 180, 180), -1)
        cv2.putText(shown, str(c), (15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # What the model actually sees, scaled up, bottom-left
    view = cv2.resize(model_view, (128, 128), interpolation=cv2.INTER_NEAREST)
    shown[-138:-10, 10:138] = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    return shown


# --- 4. Main loop -----------------------------------------------------------
def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    img = ckpt["img"]
    model = FingerNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAM}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    history = deque(maxlen=SMOOTH)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break

        x, model_view, origin = preprocess(frame, img)
        with torch.no_grad():
            history.append(torch.softmax(model(x), 1)[0].numpy())
        probs = np.mean(history, axis=0)

        cv2.imshow("live test", draw(frame, origin, model_view, probs))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
