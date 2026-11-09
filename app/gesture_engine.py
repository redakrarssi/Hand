"""MediaPipe hand tracking with relaxed, time-based scrolling intent detection."""

import logging
import math
import os
import time
import urllib.request
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from app.camera_motion import CameraMotionGuard
from app.dense_hand_tracker import DenseHandTracker
from app.touch_detector import TouchGestureDetector
from app.config import (
    MAX_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    POSITION_HISTORY_SIZE, MOTION_GESTURES_ENABLED, PALM_HOLD_ENABLED, PALM_HOLD_TIME,
    PALM_HOLD_STILL_THRESHOLD, GESTURE_COOLDOWN,
)
from app.intent_tracker import RelaxedIntentTracker

logger = logging.getLogger(__name__)
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
                    (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
                    (17, 18), (18, 19), (19, 20), (0, 17)]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info('Downloading the hand tracking model...')
        temporary = MODEL_PATH + '.download'
        try:
            urllib.request.urlretrieve(MODEL_URL, temporary)
            os.replace(temporary, MODEL_PATH)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    return MODEL_PATH


class GestureResult:
    def __init__(self, gesture_type, confidence=1.0, velocity=0.0, direction=None, source=None):
        self.type = gesture_type
        self.confidence = confidence  # Trajectory quality, not a measured accuracy probability.
        self.velocity = velocity
        self.direction = direction
        self.source = source
        self.timestamp = time.time()

    def __repr__(self):
        return f'Gesture({self.type}, source={self.source}, quality={self.confidence:.2f})'


class HandState:
    def __init__(self):
        self.intent = RelaxedIntentTracker()
        self.palm_positions = deque(maxlen=POSITION_HISTORY_SIZE)
        self.timestamps = deque(maxlen=POSITION_HISTORY_SIZE)
        self.fingers_up = [False] * 5
        self.is_open_palm = False
        self.is_fist = False
        self.smoothed_position = None
        self.palm_hold_start = None
        self.last_gesture_time = -math.inf

    def clear(self):
        self.intent.reset()
        self.palm_positions.clear()
        self.timestamps.clear()
        self.fingers_up = [False] * 5
        self.is_open_palm = False
        self.is_fist = False
        self.smoothed_position = None
        self.palm_hold_start = None

    @property
    def cycle_state(self):
        return self.intent.state


class GestureEngine:
    def __init__(self, landmarker=None, gesture_undo=False):
        if landmarker is None:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=MAX_HANDS,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
        self.landmarker = landmarker
        self.hand_state = HandState()
        self.camera_guard = CameraMotionGuard()
        self.dense_tracker = DenseHandTracker(sensitivity=1.0)
        self.touch_detector = TouchGestureDetector(sensitivity=1.0)
        from app.adaptive_learner import NoNoDetector
        self.no_no_detector = NoNoDetector(cooldown=1.4)
        self.gesture_undo = gesture_undo
        self.sensitivity = 1.0
        self.gesture_log = deque(maxlen=20)
        self._active = True
        self._current_landmarks = None
        self._dense_summary = None
        self._tracking = False
        self._camera_moving = False
        self._hand_label = None
        self._hand_label_mismatch_count = 0
        self._last_error_log = -math.inf
        self.detection_count = self.gesture_count = self.frames_processed = 0
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info('Gesture engine ready: Minimal directional tracking on important fingers')

    def set_sensitivity(self, value):
        if not math.isfinite(value):
            raise ValueError('Sensitivity must be a finite number')
        self.sensitivity = max(0.3, min(3.0, value))
        self.hand_state.intent.sensitivity = self.sensitivity
        self.dense_tracker.set_sensitivity(self.sensitivity)
        self.touch_detector.set_sensitivity(self.sensitivity)

    def apply_profile(self, profile):
        """Apply personalized user profile thresholds."""
        if hasattr(profile, 'finger_distance') and hasattr(profile, 'hand_distance'):
            self.hand_state.intent.set_learned_thresholds(
                profile.finger_distance, profile.hand_distance
            )
        if hasattr(profile, 'sensitivity') and profile.sensitivity:
            self.set_sensitivity(profile.sensitivity)
        # Apply recorded references if present in profile
        if hasattr(profile, 'gesture_references') and profile.gesture_references:
            self.load_references(profile.gesture_references)
        else:
            self.load_references()

    def load_references(self, references=None):
        """Load and apply personal 6-position reference templates."""
        try:
            from app.gesture_recorder import load_references, GestureReferences
            if references is None:
                refs = load_references()
            elif isinstance(references, dict):
                refs = GestureReferences(
                    version=references.get('version', 1),
                    recorded_at=references.get('recorded_at', 0.0),
                    is_valid=references.get('is_valid', True),
                    rest_baseline=references.get('rest_baseline'),
                    up_positions=references.get('up_positions', []),
                    down_positions=references.get('down_positions', []),
                    up_threshold=float(references.get('up_threshold', 0.020)),
                    down_threshold=float(references.get('down_threshold', 0.020)),
                    primary_channel=references.get('primary_channel', 'index'),
                    up_displacements=references.get('up_displacements', []),
                    down_displacements=references.get('down_displacements', []),
                )
            else:
                refs = references

            if refs and getattr(refs, 'is_valid', False):
                self.dense_tracker.load_references(refs)
                if hasattr(refs, 'up_threshold') and refs.up_threshold:
                    self.hand_state.intent.set_learned_thresholds(
                        refs.up_threshold, refs.up_threshold * 1.4
                    )
        except Exception as e:
            logger.warning("Could not load gesture references in engine: %s", e)

    def recalibrate(self):
        self.hand_state.clear()
        self.camera_guard.reset()
        self.dense_tracker.reset()
        self.touch_detector.reset()
        self.no_no_detector.reset()
        self._dense_summary = None
        self._hand_label = None
        self._hand_label_mismatch_count = 0

    def get_current_landmarks(self):
        return self._current_landmarks

    def get_dense_landmarks(self):
        return self.dense_tracker.points

    @staticmethod
    def _get_palm_center(landmarks):
        ids = (0, 5, 9, 13, 17)
        return (sum(landmarks[i].x for i in ids) / 5,
                sum(landmarks[i].y for i in ids) / 5)

    @staticmethod
    def _detect_fingers(landmarks):
        # Distance from wrist works for sideways hands as well as upright ones.
        wrist = landmarks[0]
        distance = lambda i: math.hypot(landmarks[i].x - wrist.x, landmarks[i].y - wrist.y)
        return [distance(tip) > distance(joint) * 1.08
                for tip, joint in ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18))]

    def _detect_swipe(self, state, landmarks, now=None, aspect=4 / 3):
        intent = state.intent.update(landmarks, time.monotonic() if now is None else now, aspect)
        if intent is None:
            return None
        state.last_gesture_time = now if now is not None else time.monotonic()
        state.palm_hold_start = None
        return GestureResult(intent.type, intent.confidence, intent.velocity, intent.direction, intent.source)

    def _detect_palm_hold(self, now):
        state = self.hand_state
        if not PALM_HOLD_ENABLED or not state.is_open_palm or state.intent.state != 'READY':
            state.palm_hold_start = None
            return None
        if now - state.last_gesture_time < GESTURE_COOLDOWN:
            return None
        positions = list(state.palm_positions)
        if len(positions) < 5 or math.dist(positions[-1], positions[0]) > PALM_HOLD_STILL_THRESHOLD:
            state.palm_hold_start = None
            return None
        if state.palm_hold_start is None:
            state.palm_hold_start = now
        elif now - state.palm_hold_start >= PALM_HOLD_TIME:
            state.palm_hold_start = None
            state.last_gesture_time = now
            return GestureResult('PALM_HOLD')
        return None

    def process_frame(self, frame_rgb, now=None, motion_frame_rgb=None):
        self.frames_processed += 1
        now = time.monotonic() if now is None else now
        if not self._active:
            self._current_landmarks = None
            self._tracking = False
            return None
        timestamp_ms = max(self._last_timestamp_ms + 1, int((now - self._start_time) * 1000))
        self._last_timestamp_ms = timestamp_ms
        try:
            result = self.landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)
        except Exception as exc:
            if now - self._last_error_log > 5:
                logger.warning('Hand tracking failed: %s', exc)
                self._last_error_log = now
            self._current_landmarks = None
            self._tracking = False
            self.hand_state.clear()
            self.camera_guard.reset()
            return None

        self._current_landmarks = result.hand_landmarks or None
        self._tracking = bool(self._current_landmarks)
        landmarks = self._current_landmarks[0] if self._tracking else None
        self._camera_moving = self.camera_guard.update(
            frame_rgb if motion_frame_rgb is None else motion_frame_rgb, landmarks)
        if not self._tracking:
            self.hand_state.intent.missing(now)
            self.hand_state.palm_hold_start = None
            self.dense_tracker.reset()
            self.touch_detector.reset()
            return None
        self.detection_count += 1
        state = self.hand_state
        center = self._get_palm_center(landmarks)
        handedness = getattr(result, 'handedness', None)
        if handedness and handedness[0] and handedness[0][0].score >= 0.80:
            label = handedness[0][0].category_name
            if self._hand_label is not None and label != self._hand_label:
                prev_center = state.smoothed_position
                jump = math.dist(center, prev_center) if prev_center else 0
                if jump > 0.05 or self._hand_label_mismatch_count >= 4:
                    self.hand_state.clear()
                    self.dense_tracker.reset()
                    self._hand_label_mismatch_count = 0
                else:
                    self._hand_label_mismatch_count += 1
            else:
                self._hand_label_mismatch_count = 0
            self._hand_label = label
        state.palm_positions.append(center)
        state.timestamps.append(now)
        if state.smoothed_position is not None:
            state.smoothed_position = (
                0.75 * center[0] + 0.25 * state.smoothed_position[0],
                0.75 * center[1] + 0.25 * state.smoothed_position[1]
            )
        else:
            state.smoothed_position = center
        state.fingers_up = self._detect_fingers(landmarks)
        state.is_open_palm = sum(state.fingers_up) >= 4
        state.is_fist = sum(state.fingers_up) == 0
        if self._camera_moving:
            return None

        touch_only = not MOTION_GESTURES_ENABLED

        # Process dense 134-point anatomical model & micro-intent
        dense_summary = self.dense_tracker.process(landmarks, frame_rgb=frame_rgb, now=now)
        self._dense_summary = dense_summary

        # Physical Finger Touch Detection:
        # Big finger (Thumb) touching Index -> Scroll DOWN
        # Big finger (Thumb) touching Middle -> Scroll UP
        touch_gesture = self.touch_detector.update(landmarks, now=now)
        if touch_gesture:
            self.gesture_count += 1
            self.gesture_log.append(touch_gesture)
            state.last_gesture_time = now
            state.intent.stroke = []
            return touch_gesture

        # While pinching, closing in on a pinch, or just after one, the finger
        # motion is part of the touch gesture - never also a swipe/micro scroll.
        if self.touch_detector.is_engaged(now):
            state.intent.stroke = []
            return None

        # Moving the hand must never scroll: only the touch gesture above does.
        if touch_only:
            state.intent.update(landmarks, now, frame_rgb.shape[1] / frame_rgb.shape[0])
            state.intent.stroke = []
            return None

        # Sideways movements are ordinary repositioning unless undo was enabled.
        if self.gesture_undo and self.no_no_detector.update(landmarks, now):
            self.gesture_count += 1
            res = GestureResult('NO_NO', confidence=1.0, velocity=0.0, source='wag')
            self.gesture_log.append(res)
            return res
        gesture = self._detect_swipe(state, landmarks, now, frame_rgb.shape[1] / frame_rgb.shape[0])
        if gesture is None:
            gesture = self._detect_palm_hold(now)
        # Instant dense micro-gesture trigger (zero delay, under 40ms)
        if gesture is None and state.intent.state in ('READY', 'MOVING') and dense_summary and dense_summary.get('micro_gesture'):
            mg = dense_summary['micro_gesture']
            gesture = GestureResult(
                mg['type'],
                confidence=mg['confidence'],
                velocity=mg['velocity'],
                direction=(0, 1 if mg['type'] == 'SWIPE_DOWN' else -1),
                source=mg['source']
            )
            state.last_gesture_time = now
            # Synchronize intent state to RECOVERING so recoil/return movement is swallowed immediately
            state.intent.state = 'RECOVERING'
            state.intent.recovery_origin = state.intent.baseline or (state.intent.history[0] if state.intent.history else None)
            state.intent.recovery_channel = 0 if 'index' in mg['source'] else (1 if 'middle' in mg['source'] else 4)
            state.intent.last_fire = now
            state.intent.stroke = []
        if gesture:
            self.gesture_count += 1
            self.gesture_log.append(gesture)
        return gesture

    def get_hand_info(self):
        state = self.hand_state
        dense = self._dense_summary or self.dense_tracker.get_summary()
        is_resting = self.dense_tracker.is_resting and state.intent.state == 'READY'
        touch_info = self.touch_detector.get_telemetry()

        if touch_info.get('touching_index'):
            status = '[TOUCH] THUMB + INDEX: SCROLL DOWN'
        elif touch_info.get('touching_middle'):
            status = '[TOUCH] THUMB + MIDDLE: SCROLL UP'
        elif self._camera_moving:
            status = 'CAMERA MOVED - HOLD STILL'
        elif not self._tracking:
            status = 'SHOW HAND AND KNUCKLES IN VIEW'
        elif MOTION_GESTURES_ENABLED and dense.get('micro_intent') == 'UP':
            status = '[MICRO-SCROLL] TOP'
        elif MOTION_GESTURES_ENABLED and dense.get('micro_intent') == 'DOWN':
            status = '[MICRO-SCROLL] BOTTOM'
        elif not MOTION_GESTURES_ENABLED:
            status = 'READY - TOUCH THUMB TO INDEX / MIDDLE'
        elif is_resting:
            stability_pct = int(self.dense_tracker.resting_stability * 100)
            status = f'[RESTING] (STABLE {stability_pct}%)'
        else:
            status = {
                'CALIBRATING': 'RELAX HAND TO CALIBRATE',
                'READY': 'READY - TOUCH OR SMALL STROKE',
                'MOVING': f'READING {(state.intent.source or "hand").upper()}',
                'RECOVERING': 'RELAX HAND TO REARM',
            }.get(state.intent.state, 'READY')

        return {
            'position': state.smoothed_position,
            'fingers_up': state.fingers_up,
            'is_open_palm': state.is_open_palm,
            'is_fist': state.is_fist,
            'is_resting': is_resting,
            'cycle_state': state.intent.state,
            'intent_status': status,
            'intent_progress': state.intent.progress,
            'motion_source': state.intent.source or dense.get('micro_intent'),
            'finger_count': sum(state.fingers_up),
            'tracking': self._tracking,
            'detection_rate': self.detection_count / max(1, self.frames_processed) * 100,
            # Dense 134-point tracking metrics:
            'dense_landmarks': self.dense_tracker.points,
            'dense_connections': self.dense_tracker.connections,
            'dense_summary': dense,
            'resting_stability': self.dense_tracker.resting_stability,
            'micro_intent': dense.get('micro_intent'),
            'total_points': self.dense_tracker.point_count,
            'dominant_points': self.dense_tracker.dominant_point_count,
            'noise_floor': self.dense_tracker.noise_floor,
            # Touch gesture metrics:
            'touch_active': touch_info.get('active_touch'),
            'touching_index': touch_info.get('touching_index', False),
            'touching_middle': touch_info.get('touching_middle', False),
            'dist_thumb_index': touch_info.get('dist_thumb_index', 1.0),
            'dist_thumb_middle': touch_info.get('dist_thumb_middle', 1.0),
            'thumb_pos': touch_info.get('thumb_pos', (0.0, 0.0)),
            'index_pos': touch_info.get('index_pos', (0.0, 0.0)),
            'middle_pos': touch_info.get('middle_pos', (0.0, 0.0)),
            'touch_threshold': touch_info.get('touch_threshold', 0.28),
            'release_threshold': touch_info.get('release_threshold', 0.38),
        }

    def get_stats(self):
        return {'frames_processed': self.frames_processed, 'detections': self.detection_count,
                'gestures': self.gesture_count,
                'detection_rate': f'{self.detection_count / max(1, self.frames_processed) * 100:.1f}%',
                'sensitivity': self.sensitivity}

    def pause(self):
        self._active = False
        self.recalibrate()

    def resume(self):
        self.recalibrate()
        self._active = True

    def is_active(self):
        return self._active

    def cleanup(self):
        self.landmarker.close()
