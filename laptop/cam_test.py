"""Webcam sanity check for the finger-count lock box.

    python laptop/cam_test.py --list      # find which camera indices work
    python laptop/cam_test.py --cam 2     # live preview; s = save frame, q = quit
"""
import argparse
import time
from pathlib import Path

import cv2

FRAMES_DIR = Path(__file__).parent / "frames"


# --- 1. Probe cameras -------------------------------------------------------
def list_cameras(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            print(f"[{i}] OK  {w}x{h}")
        else:
            print(f"[{i}] no frames")


# --- 2. Live preview --------------------------------------------------------
def preview(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    FRAMES_DIR.mkdir(exist_ok=True)
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break

        # Smoothed FPS so the number doesn't flicker
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        shown = frame.copy()
        cv2.putText(shown, f"{fps:.1f} fps  {frame.shape[1]}x{frame.shape[0]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"cam {index}", shown)

        # --- 3. Keys ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = FRAMES_DIR / f"frame_{int(now * 1000)}.jpg"
            cv2.imwrite(str(path), frame)  # save the clean frame, no overlay
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


# --- 4. CLI -----------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="probe camera indices")
    p.add_argument("--cam", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    if args.list:
        list_cameras()
    else:
        preview(args.cam, args.width, args.height)
