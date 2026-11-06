"""Collect labeled finger-count images from a fixed on-screen box.

    python laptop/collect.py --cam 5

Keys: 0-5 = start/stop recording that class (0 = fist / no hand), q = quit.
Images go to laptop/data/<session>/<class>/<n>.png (64x64 grayscale).
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).parent / "data"
CLASSES = [str(c) for c in range(6)]


# --- 1. Box + crop ----------------------------------------------------------
def box_coords(frame, size):
    """Top-left and bottom-right of a size x size box centered in the frame."""
    h, w = frame.shape[:2]
    x0, y0 = (w - size) // 2, (h - size) // 2
    return x0, y0, x0 + size, y0 + size


def crop(frame, box, out_size):
    x0, y0, x1, y1 = box
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)


# --- 2. Overlay -------------------------------------------------------------
def draw(frame, box, recording, counts):
    shown = cv2.flip(frame, 1)  # mirror for display only; saved crops are not flipped
    x0, y0, x1, y1 = box
    w = frame.shape[1]
    color = (0, 0, 255) if recording is not None else (0, 255, 0)
    cv2.rectangle(shown, (w - x1, y0), (w - x0, y1), color, 2)

    status = f"REC class {recording}" if recording is not None else "idle (press 0-5)"
    cv2.putText(shown, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    line = "  ".join(f"{c}:{counts[c]}" for c in CLASSES)
    cv2.putText(shown, line, (10, shown.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return shown


# --- 3. Main loop -----------------------------------------------------------
def main(cam, box_size, out_size, rate):
    session = DATA_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    for c in CLASSES:
        (session / c).mkdir(parents=True, exist_ok=True)
    print(f"Session: {session}")

    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counts = {c: 0 for c in CLASSES}
    recording = None
    last_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break
        box = box_coords(frame, box_size)

        # Save at most `rate` images/sec so neighbors aren't near-duplicates
        now = time.time()
        if recording is not None and now - last_save >= 1.0 / rate:
            img = crop(frame, box, out_size)
            cv2.imwrite(str(session / recording / f"{counts[recording]:04d}.png"), img)
            counts[recording] += 1
            last_save = now

        cv2.imshow("collect", draw(frame, box, recording, counts))

        # --- 4. Keys ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        ch = chr(key) if key < 128 else ""
        if ch in CLASSES:
            recording = None if recording == ch else ch
            print(f"Recording: {recording}")

    cap.release()
    cv2.destroyAllWindows()
    print("Saved:", counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=5)
    p.add_argument("--box", type=int, default=224, help="box size in camera pixels")
    p.add_argument("--size", type=int, default=64, help="saved image size")
    p.add_argument("--rate", type=float, default=5.0, help="images saved per second")
    args = p.parse_args()
    main(args.cam, args.box, args.size, args.rate)
