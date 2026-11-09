"""
Measure real camera noise and moving_pixels on camera 0.
"""
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open camera 0")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_gray = None
noise_readings = []

print("Capturing 30 frames from camera 0 to measure noise floor...")
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        continue
    small = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        for thresh_val in [15, 20, 25, 30]:
            _, th = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            pixels = int(np.sum(th > 0))
            if thresh_val == 25:
                noise_readings.append(pixels)
    prev_gray = gray
    time.sleep(0.033)

cap.release()

if noise_readings:
    print(f"Noise floor (threshold=25): min={min(noise_readings)}, max={max(noise_readings)}, avg={sum(noise_readings)/len(noise_readings):.1f}")
    print(f"Readings: {noise_readings[:15]}")
