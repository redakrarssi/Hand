import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import cv2
from app.gesture_engine import GestureEngine

print("Starting camera for 5 seconds to test touch detection...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

engine = GestureEngine()
start = time.time()
frame_idx = 0

while time.time() - start < 5.0:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = time.monotonic()
    gesture = engine.process_frame(frame_rgb, now=now)
    info = engine.get_hand_info()
    if info.get('tracking'):
        d_ti = info.get('dist_thumb_index', 0.0)
        d_tm = info.get('dist_thumb_middle', 0.0)
        t_idx = info.get('touching_index')
        t_mid = info.get('touching_middle')
        print(f"[{frame_idx:03d}] d_ti={d_ti:.3f} d_tm={d_tm:.3f} touching_idx={t_idx} touching_mid={t_mid} gesture={gesture}")
    else:
        if frame_idx % 10 == 0:
            print(f"[{frame_idx:03d}] Searching for hand...")
    frame_idx += 1
    time.sleep(0.033)

cap.release()
print("Done test.")
