"""Time-based, palm-size-normalized intent tracking, independent of the camera.

A movement is evidence of intent, not proof of it. Ambiguous trajectories are
discarded. Each finger has its own trajectory; a finger switch cannot invent
velocity. No open-palm pose or fast per-frame jump is required.
"""

import math
from collections import deque
from dataclasses import dataclass
from statistics import median

from app.config import (
    GESTURE_COOLDOWN, INTENT_CALIBRATION_TIME, INTENT_SETTLE_TIME,
    INTENT_MIN_STROKE_TIME, INTENT_MAX_STROKE_TIME, INTENT_FINGER_DISTANCE,
    INTENT_HAND_DISTANCE, INTENT_DIRECTION_RATIO, INTENT_TRACKING_GAP,
    INTENT_RECOVERY_TIMEOUT,
)

CHANNELS = ((8, 5, 'index'), (12, 9, 'middle'), (16, 13, 'ring'),
            (20, 17, 'little'), (4, 2, 'thumb'))
SOURCES = tuple(item[2] for item in CHANNELS) + ('hand',)


def bounded_threshold(value, default, minimum, maximum):
    # Old calibration could confuse posture changes with sensor noise, saving
    # thresholds of several palm lengths. Keep small strokes within reach.
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return default
    if value > maximum * 4:
        return default
    return max(minimum, min(maximum, value))


@dataclass
class Sample:
    time: float
    points: tuple
    scale: float


@dataclass
class Intent:
    type: str
    confidence: float
    velocity: float
    direction: tuple
    source: str


class RelaxedIntentTracker:
    def __init__(self, sensitivity=1.0, finger_distance=None, hand_distance=None):
        self.sensitivity = sensitivity
        self.finger_distance = bounded_threshold(finger_distance, INTENT_FINGER_DISTANCE, 0.018, 0.065)
        self.hand_distance = bounded_threshold(hand_distance, INTENT_HAND_DISTANCE, 0.028, 0.085)
        self.reset()

    def set_learned_thresholds(self, finger_distance, hand_distance):
        self.finger_distance = bounded_threshold(finger_distance, INTENT_FINGER_DISTANCE, 0.018, 0.065)
        self.hand_distance = bounded_threshold(hand_distance, INTENT_HAND_DISTANCE, 0.028, 0.085)
        self.reset()

    def reset(self):
        self.state = 'CALIBRATING'
        self.baseline = None
        self.history = deque(maxlen=240)
        self.raw_history = deque(maxlen=3)
        self.stroke = []
        self.noise = [0.001] * len(SOURCES)
        self.last_seen = None
        self.last_fire = -math.inf
        self.source = None
        self.progress = 0.0
        self.recovery_channel = None
        self.recovery_origin = None

    def missing(self, now):
        if self.last_seen is not None and now - self.last_seen > INTENT_TRACKING_GAP:
            self.reset()

    @staticmethod
    def extract(landmarks, now, aspect=4 / 3):
        if len(landmarks) != 21:
            return None
        if any(not math.isfinite(v) for p in landmarks for v in (p.x, p.y)):
            return None
        palm_ids = (0, 5, 9, 13, 17)
        if any(not (-0.25 <= landmarks[i].x <= 1.25 and
                    -0.25 <= landmarks[i].y <= 1.25) for i in palm_ids):
            return None
        palm = (sum(landmarks[i].x for i in palm_ids) * aspect / 5,
                sum(landmarks[i].y for i in palm_ids) / 5)
        wrist = landmarks[0]
        scale = median(math.hypot((landmarks[i].x - wrist.x) * aspect,
                                 landmarks[i].y - wrist.y) for i in palm_ids[1:])
        if scale < 0.015 or scale > 0.85:
            return None
        points = tuple(((landmarks[tip].x - landmarks[base].x) * aspect,
                        landmarks[tip].y - landmarks[base].y)
                       for tip, base, _ in CHANNELS) + (palm,)
        return Sample(now, points, scale)

    def _threshold(self, channel):
        scale = self.baseline.scale
        distance = self.hand_distance if channel == 5 else self.finger_distance
        # A noise floor stays in force even at the highest sensitivity.
        pixel_floor = 0.004 if channel == 5 else 0.0025
        # Cap noise contribution so noisy camera sensors do not push threshold out of reach
        noise_part = min(distance * 1.3, self.noise[channel] * 3.5 / scale)
        return max(distance / self.sensitivity, noise_part, pixel_floor / scale)

    def _recent(self, now, duration):
        if len(self.history) >= 2:
            duration = max(duration, (self.history[-1].time - self.history[-2].time) * 2.1)
        return [s for s in self.history if now - s.time <= duration + 0.001]

    def _quiet(self, samples, channel, threshold):
        if len(samples) < 3 or samples[-1].time - samples[0].time < INTENT_SETTLE_TIME * 0.70:
            return False
        scale = self.baseline.scale if self.baseline else samples[-1].scale
        tolerance = max(threshold * 0.25, self.noise[channel] * 2.5 / scale, 0.008)
        # Check vertical axis for vertical stroke settling
        values = [s.points[channel][1] / scale for s in samples]
        if max(values) - min(values) > tolerance:
            return False
        return True

    def _calibrate(self, sample):
        recent = self._recent(sample.time, INTENT_CALIBRATION_TIME)
        if len(recent) < 3 or recent[-1].time - recent[0].time < INTENT_CALIBRATION_TIME * 0.75:
            return
        # Acquire a relaxed posture only after the hand has stopped entering view.
        # Allows natural resting breathing / resting micro-movement during calibration.
        stable = []
        for channel in range(len(SOURCES)):
            thresh = sample.scale * 0.075 if channel == 5 else max(0.007, sample.scale * 0.075)
            stable.append(all(
                max(s.points[channel][axis] for s in recent) -
                min(s.points[channel][axis] for s in recent) <= thresh
                for axis in (0, 1)))
        # A steady palm plus one steady finger is sufficient. Other relaxed
        # fingers can move slightly without restarting acquisition forever.
        if not stable[5] or not any(stable[:5]):
            return
        points = tuple(tuple(median(s.points[c][a] for s in recent) for a in (0, 1))
                       for c in range(len(SOURCES)))
        self.baseline = Sample(sample.time, points, median(s.scale for s in recent))
        for c in range(len(SOURCES)):
            deviations = [abs(s.points[c][1] - points[c][1]) for s in recent]
            self.noise[c] = min(0.012 * sample.scale, max(0.0005, median(deviations) * 1.4826))
        self.state = 'READY'

    def update(self, landmarks, now, aspect=4 / 3):
        raw = self.extract(landmarks, now, aspect)
        if raw is None:
            self.reset()
            return None
        if self.last_seen is not None:
            if now <= self.last_seen:
                return None
            if now - self.last_seen > INTENT_TRACKING_GAP:
                self.reset()
        self.last_seen = now
        self.raw_history.append(raw)
        # Three-frame median removes isolated landmark spikes without mixing fingers.
        points = tuple(tuple(median(s.points[c][a] for s in self.raw_history)
                             for a in (0, 1)) for c in range(len(SOURCES)))
        sample = Sample(now, points, median(s.scale for s in self.raw_history))
        self.history.append(sample)
        if self.baseline is None:
            self._calibrate(sample)
            return None

        scale = self.baseline.scale
        scale_ratio = sample.scale / scale
        if abs(scale_ratio - 1) > 0.55:
            self.reset()  # Hand changed or large depth jump.
            return None
        # Smoothly adapt scale for natural distance/tilt variation
        if abs(scale_ratio - 1) > 0.08:
            self.baseline.scale = 0.90 * self.baseline.scale + 0.10 * sample.scale
            scale = self.baseline.scale

        deltas = [(p[1] - b[1]) / scale for p, b in zip(points, self.baseline.points)]
        scores = [abs(dy) / self._threshold(c) for c, dy in enumerate(deltas)]
        channel = max(range(len(scores)), key=scores.__getitem__)
        self.progress = min(1.0, scores[channel])

        if self.state == 'RECOVERING':
            c = self.recovery_channel if self.recovery_channel is not None else 0
            origin = self.recovery_origin or self.baseline or (self.history[0] if self.history else sample)
            returned = abs(points[c][1] - origin.points[c][1]) / scale < self._threshold(c) * 0.85
            quiet = self._quiet(self._recent(now, INTENT_RECOVERY_TIMEOUT), c, self._threshold(c))
            if returned or (now - self.last_fire > INTENT_RECOVERY_TIMEOUT and quiet):
                # Returning to neutral rearms immediately, even during cooldown.
                # Otherwise a quick next stroke can leave neutral before the
                # cooldown ends and be mistaken for the previous stroke's return.
                # Entering the neutral band is not a new resting position: using
                # its edge as neutral would accumulate drift across fast repeats.
                self.baseline = (Sample(now, origin.points, scale)
                                 if returned else sample)
                self.state = 'READY'
                self.source = None
                self.progress = 0.0
                # Never seed the next stroke with a consumed outward/return path.
                self.history.clear()
                self.history.append(sample)
                self.stroke = []
            return None

        if self.state == 'READY':
            if scores[channel] >= 0.45:
                self.state = 'MOVING'
                self.source = SOURCES[channel]
                # Preserve the first half of small/slow strokes instead of losing it.
                recent = self._recent(now, 0.35)
                start = 0
                for i, s in enumerate(recent[:-1]):
                    if abs(s.points[channel][1] - self.baseline.points[channel][1]) / scale < self._threshold(channel) * 0.15:
                        start = i
                self.stroke = [Sample(recent[start].time, self.baseline.points, scale)] + recent[start + 1:]
            else:
                # Follow horizontal repositioning anywhere in the camera view.
                # Otherwise the old x baseline makes later vertical strokes
                # look diagonal, effectively trapping the hand in one spot.
                moved_sideways = abs(points[5][0] - self.baseline.points[5][0]) > scale * 0.10
                neutral_x = tuple((p[0], b[1]) for p, b in zip(points, self.baseline.points))
                self.baseline = Sample(self.baseline.time, neutral_x, scale)
                if moved_sideways:
                    self.history.clear()
                    self.history.append(sample)
                # Adapt only inside a small neutral zone. Slow intentional travel
                # outside this zone accumulates against a fixed resting baseline.
                if max(scores) < 0.22:
                    dt = sample.time - self.baseline.time
                    alpha = min(0.035, max(0, dt) * 0.025)
                    neutral = tuple(tuple(b[a] + alpha * (p[a] - b[a]) for a in (0, 1))
                                    for b, p in zip(self.baseline.points, points))
                    self.baseline = Sample(now, neutral, scale)
            return None

        self.stroke.append(sample)
        if now - self.stroke[0].time > INTENT_MAX_STROKE_TIME:
            self.baseline = sample
            self.state = 'READY'
            self.stroke = []
            self.source = None
            return None

        # Use each channel's full path, including preparation and return stroke.
        candidates = self._evaluate()
        if not candidates:
            # A sub-threshold movement that stopped is just a new resting position.
            if all(self._quiet(self._recent(now, max(0.20, INTENT_SETTLE_TIME)), c, self._threshold(c))
                   for c in range(len(SOURCES))):
                self.baseline = sample
                self.state = 'READY'
                self.stroke = []
                self.source = None
            return None

        best = max(candidates, key=lambda candidate: candidate[0])
        _, c, sign, peak, coherence, duration, ready_to_fire = best
        self.source = SOURCES[c]
        if not ready_to_fire:
            return None

        # Conflicting, similarly strong finger/hand signals have uncertain intent.
        if any(other[2] != sign and other[0] > best[0] * 0.75 for other in candidates):
            self.baseline = sample
            self.state = 'READY'
            self.stroke = []
            self.source = None
            return None

        # Keep collecting the new stroke during cooldown; gate only emission.
        if now - self.last_fire < GESTURE_COOLDOWN:
            return None

        self.recovery_origin = self.baseline
        self.recovery_channel = c
        self.baseline = sample
        self.last_fire = now
        self.state = 'RECOVERING'
        self.stroke = []
        return Intent('SWIPE_DOWN' if sign > 0 else 'SWIPE_UP',
                      min(0.98, 0.60 + 0.25 * coherence + min(peak / self._threshold(c) - 1, 1) * 0.10),
                      peak * scale / max(duration, 0.001), (0, sign * peak * scale), SOURCES[c])

    def _evaluate(self):
        if len(self.stroke) < 3:
            return []
        scale = self.baseline.scale
        candidates = []
        now = self.stroke[-1].time
        for c in range(len(SOURCES)):
            base_x, base_y = self.baseline.points[c]
            ys = [(s.points[c][1] - base_y) / scale for s in self.stroke]
            xs = [(s.points[c][0] - base_x) / scale for s in self.stroke]
            up, down = max(0, -min(ys)), max(0, max(ys))
            threshold = self._threshold(c)
            min_y = min(ys)
            max_y = max(ys)
            up = max(0, -min_y)
            down = max(0, max_y)

            if up < threshold * 0.70 and down < threshold * 0.70 and (up + down) < threshold:
                continue

            idx_up = min(range(len(ys)), key=lambda i: ys[i])
            idx_down = max(range(len(ys)), key=lambda i: ys[i])
            has_real_up = up >= threshold * 0.35
            has_real_down = down >= threshold * 0.35

            if has_real_up and has_real_down:
                if idx_up < idx_down:
                    # Upward excursion occurred BEFORE downward excursion.
                    downward_travel = down + up
                    is_down_with_prep_lift = (
                        down >= threshold * 0.45 and
                        downward_travel >= threshold and
                        (downward_travel >= up * 1.10 or down >= up * 0.65)
                    )
                    if is_down_with_prep_lift:
                        sign = 1
                        peak = max(down, downward_travel * 0.75)
                        peak_index = idx_down
                    else:
                        sign = -1
                        peak = up
                        peak_index = idx_up
                else:
                    # Downward excursion occurred BEFORE upward excursion.
                    upward_travel = up + down
                    is_up_with_prep_dip = (
                        up >= threshold * 0.45 and
                        upward_travel >= threshold and
                        (upward_travel >= down * 1.10 or up >= down * 0.65)
                    )
                    if is_up_with_prep_dip:
                        sign = -1
                        peak = max(up, upward_travel * 0.75)
                        peak_index = idx_up
                    else:
                        sign = 1
                        peak = down
                        peak_index = idx_down
            elif has_real_up:
                sign = -1
                peak = up
                peak_index = idx_up
            else:
                sign = 1
                peak = down
                peak_index = idx_down

            if peak < threshold:
                continue

            peak_start = 0
            if sign == 1 and idx_up < idx_down and (down + up) >= threshold:
                peak_start = idx_up
            elif sign == -1 and idx_down < idx_up and (up + down) >= threshold:
                peak_start = idx_down

            duration = self.stroke[peak_index].time - self.stroke[peak_start].time
            if duration < INTENT_MIN_STROKE_TIME:
                continue

            if peak < max(abs(x) for x in xs[:peak_index + 1]) * INTENT_DIRECTION_RATIO:
                continue

            # Trajectory coherence along active stroke travel
            path = sum(abs(ys[i] - ys[i - 1]) + 0.3 * abs(xs[i] - xs[i - 1])
                       for i in range(peak_start + 1, peak_index + 1))
            coherence = peak / max(path, peak)
            if coherence < 0.30:
                continue

            current_disp = ys[-1] * sign
            reverse_margin = max(threshold * 0.10, self.noise[c] * 2.0 / scale)
            has_reversed = peak_index < len(ys) - 1 and (peak - current_disp) >= reverse_margin
            is_quiet = self._quiet(self._recent(now, INTENT_SETTLE_TIME), c, threshold)
            returned = abs(ys[-1]) < threshold * 0.60

            raw_dy = 0.0
            if len(self.raw_history) >= 2:
                raw_dy = (self.raw_history[-1].points[c][1] - self.raw_history[-2].points[c][1]) / scale

            # Immediate action 1ms after stroke completion:
            stopped_at_peak = (len(ys) >= 3 and
                               abs(peak - current_disp) < reverse_margin * 1.2 and
                               abs(raw_dy) < 0.003 and
                               abs(ys[-1] - ys[-2]) < 0.003 and
                               peak >= threshold)
            decisive = peak >= threshold * 1.30 and coherence >= 0.75 and duration >= 0.18

            # Biomechanical counter-movement flight guards:
            # When a tentative peak reverses, but the hand/finger is in active rapid flight
            # in the opposite direction towards or through baseline, hold fire!
            # It will either cross baseline into a full counter-stroke (prep lift -> DOWN, prep dip -> UP),
            # or settle near baseline (in which case it will fire upon slowing/settling).
            in_downward_flight = (sign == -1 and peak_index < len(ys) - 1 and
                                  raw_dy > 0.003 and
                                  (peak - current_disp >= reverse_margin * 0.4))
            in_upward_flight = (sign == 1 and peak_index < len(ys) - 1 and
                                raw_dy < -0.003 and
                                (peak - current_disp >= reverse_margin * 0.4))

            if in_downward_flight or in_upward_flight:
                ready_to_fire = False
            else:
                ready_to_fire = decisive or has_reversed or stopped_at_peak or is_quiet or returned

            candidates.append((peak / threshold * coherence, c, sign, peak, coherence, duration, ready_to_fire))
        return candidates
