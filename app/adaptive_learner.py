"""Optional side-to-side gesture undo; direction and sensitivity stay fixed."""

import math
import time
import logging
from collections import deque

logger = logging.getLogger("HandScroller.AdaptiveLearner")


class NoNoDetector:
    """
    Detects the 'no-no' gesture:
    - Index finger wagging side-to-side (left-right-left)
    - Or whole hand waving side-to-side (shake left-right-left)
    Used only for optional undo of a recent action. Direction and sensitivity
    remain controlled by explicit user settings.
    """

    def __init__(self, cooldown=1.4):
        self.cooldown = cooldown
        self.last_detected = -math.inf
        self.history = deque(maxlen=45)  # ~1.5s at 30 FPS

    def reset(self):
        self.history.clear()

    def update(self, landmarks, now):
        """
        Feed landmarks. Returns True if a 'no-no' gesture was recognized on this frame.
        """
        if now - self.last_detected < self.cooldown:
            return False

        if not landmarks:
            return False

        # Unpack 2D list if needed
        if isinstance(landmarks, (list, tuple)) and len(landmarks) > 0 and isinstance(landmarks[0], (list, tuple)):
            landmarks = landmarks[0]

        if len(landmarks) < 21:
            return False

        # Extract wrist (0), middle knuckle (9), index tip (8), index base (5)
        wrist = landmarks[0]
        knuckle = landmarks[9]
        index_tip = landmarks[8]
        index_base = landmarks[5]

        scale = max(0.018, math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y))

        tip_x = index_tip.x
        tip_y = index_tip.y
        palm_x = (wrist.x + knuckle.x) / 2.0
        palm_y = (wrist.y + knuckle.y) / 2.0
        finger_wag_x = (index_tip.x - index_base.x) / scale

        self.history.append((now, tip_x, tip_y, palm_x, palm_y, finger_wag_x, scale))

        if len(self.history) < 8:
            return False

        # Consider the last 0.85 seconds of history
        recent = [s for s in self.history if now - s[0] <= 0.85]
        if len(recent) < 7:
            return False

        # Check either relative finger wag or palm wave oscillation
        if self._check_oscillation(recent, idx_val=5, min_amp=0.20) or \
           self._check_oscillation(recent, idx_val=1, min_amp=0.035) or \
           self._check_oscillation(recent, idx_val=3, min_amp=0.030):
            self.last_detected = now
            self.history.clear()
            return True

        return False

    def _check_oscillation(self, samples, idx_val, min_amp):
        """
        Check if series in samples[idx_val] exhibits >= 2 direction reversals (left-right-left).
        """
        values = [s[idx_val] for s in samples]
        times = [s[0] for s in samples]

        # 3-sample moving average smoothing to eliminate landmark camera noise
        smoothed = []
        for i in range(len(values)):
            sub = values[max(0, i - 1):min(len(values), i + 2)]
            smoothed.append(sum(sub) / len(sub))

        total_range = max(smoothed) - min(smoothed)
        if total_range < min_amp:
            return False

        # Ensure horizontal motion dominates vertical motion
        if idx_val in (1, 3):
            y_idx = idx_val + 1
            y_range = max(s[y_idx] for s in samples) - min(s[y_idx] for s in samples)
            if y_range > total_range * 1.55:
                return False  # Vertical motion dominated; not a side-to-side wag

        # Find local peaks and valleys
        reversals = []
        for i in range(1, len(smoothed) - 1):
            prev_v = smoothed[i - 1]
            curr_v = smoothed[i]
            next_v = smoothed[i + 1]

            if curr_v > prev_v and curr_v >= next_v:
                reversals.append((times[i], curr_v, 'peak'))
            elif curr_v < prev_v and curr_v <= next_v:
                reversals.append((times[i], curr_v, 'valley'))

        if len(reversals) < 3:
            return False

        # Filter out tiny micro-reversals
        significant_reversals = [reversals[0]]
        for r in reversals[1:]:
            last_r = significant_reversals[-1]
            if r[2] != last_r[2]:  # Alternating peak/valley
                amp = abs(r[1] - last_r[1])
                dt = r[0] - last_r[0]
                if amp >= min_amp * 0.38 and 0.05 <= dt <= 0.45:
                    significant_reversals.append(r)
            else:
                if (r[2] == 'peak' and r[1] > last_r[1]) or (r[2] == 'valley' and r[1] < last_r[1]):
                    significant_reversals[-1] = r

        return len(significant_reversals) >= 3


class AdaptiveLearner:
    """Optional gesture undo. Hand movements never retune user preferences."""

    def __init__(self):
        self.no_no_detector = NoNoDetector(cooldown=1.4)
        self.last_action = None
        self.action_history = deque(maxlen=10)

    def record_action(self, action_info, gesture):
        if not action_info or not action_info.get('success'):
            return
        entry = {
            'time': time.monotonic(),
            # Keep the live record: smooth scrolling updates the amount actually
            # delivered. Copying this dict would undo only its first wheel step.
            'action_info': action_info,
            'gesture_type': getattr(gesture, 'type', ''),
        }
        self.last_action = entry
        self.action_history.append(entry)

    def process_frame(self, landmarks, now):
        return self.no_no_detector.update(landmarks, now)

    def handle_correction(self, gesture_engine, input_sim):
        """Undo one recent action without guessing what the user wanted."""
        action = self.last_action
        self.last_action = None
        if not action or time.monotonic() - action['time'] > 2.8:
            return {'undone': False, 'type': 'none',
                    'description': 'No recent scroll to undo', 'detail': ''}
        undone = input_sim.undo_action(action['action_info'])
        return {'undone': undone, 'type': 'undo' if undone else 'none',
                'description': 'Last scroll undone' if undone else 'Undo unavailable - focus the original page',
                'detail': 'Direction and sensitivity remain as you set them'}
