"""Reject camera movement using tracked background points outside the hand."""

import cv2
import numpy as np


class CameraMotionGuard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_bounds = None

    def update(self, rgb, landmarks):
        gray = cv2.cvtColor(cv2.resize(rgb, (160, 120)), cv2.COLOR_RGB2GRAY)
        bounds = None
        if landmarks:
            xs = [p.x * 160 for p in landmarks]
            ys = [p.y * 120 for p in landmarks]
            # Generously mask the hand and the forearm extending towards the bottom edge
            bounds = (max(0, int(min(xs)) - 32), max(0, int(min(ys)) - 32),
                      min(160, int(max(xs)) + 32), 120)
        moving = False
        if self.previous is not None:
            mask = np.full_like(gray, 255)
            for box in (bounds, self.previous_bounds):
                if box is not None:
                    x1, y1, x2, y2 = box
                    mask[y1:y2, x1:x2] = 0
            corners = cv2.goodFeaturesToTrack(self.previous, 60, 0.03, 8, mask=mask)
            if corners is not None and len(corners) >= 12:
                tracked, status, errors = cv2.calcOpticalFlowPyrLK(
                    self.previous, gray, corners, None, winSize=(15, 15), maxLevel=2)
                if tracked is not None:
                    good = (status.reshape(-1) == 1) & (errors.reshape(-1) < 25)
                    offsets = (tracked - corners).reshape(-1, 2)[good]
                    if len(offsets) >= 12:
                        shift = np.median(offsets, axis=0)
                        agreement = np.mean(np.linalg.norm(offsets - shift, axis=1) < 1.0)
                        good_pts = corners[good].reshape(-1, 2)
                        spread_x = np.ptp(good_pts[:, 0]) if len(good_pts) > 0 else 0
                        spread_y = np.ptp(good_pts[:, 1]) if len(good_pts) > 0 else 0
                        is_broad = spread_x > 50 and spread_y > 35
                        moving = bool(np.linalg.norm(shift) > 2.2 and agreement > 0.75 and is_broad)
            # Large exposure changes are also unreliable tracking evidence.
            background = mask > 0
            if np.count_nonzero(background) > gray.size * 0.25:
                diff = cv2.absdiff(gray, self.previous)[background]
                moving = moving or bool(np.median(diff) > 45)
        self.previous = gray
        self.previous_bounds = bounds
        return moving
