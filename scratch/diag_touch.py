"""Live pinch diagnostic: python scratch/diag_touch.py  (pinch thumb+index, then thumb+middle, for 15s)"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from app.gesture_engine import GestureEngine
from app.input_simulator import _ensure_cursor_over_target, _get_foreground_handle

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
eng = GestureEngine()
t0 = time.monotonic(); frames = tracked = moving = 0
mn_ti = mn_tm = 9; events = []
while time.monotonic() - t0 < 15:
    ok, f = cap.read()
    if not ok: continue
    f = cv2.flip(f, 1)
    g = eng.process_frame(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), now=time.monotonic())
    frames += 1
    tracked += eng._tracking; moving += eng._camera_moving
    if eng._tracking:
        mn_ti = min(mn_ti, eng.touch_detector.dist_thumb_index); mn_tm = min(mn_tm, eng.touch_detector.dist_thumb_middle)
    if g: events.append((round(time.monotonic()-t0,2), g.type, g.source))
print(f"frames={frames} hand_tracked={tracked} camera_moving_frames={moving}")
print(f"min thumb-index={mn_ti:.3f} min thumb-middle={mn_tm:.3f} touch_threshold={eng.touch_detector.touch_threshold:.3f}")
print("events:", events)
print("cursor over foreground window:", _ensure_cursor_over_target(_get_foreground_handle()))
