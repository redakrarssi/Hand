"""
Gesture Recorder Module - Whole Movement Trajectory Reference Recorder.

Guides the user through recording:
1. Resting Neutral Baseline (calibrates rest pose and noise floor)
2. Three WHOLE movements for Scroll UP:
   - Movement 1: Slight / Micro upward flick or stroke
   - Movement 2: Medium / Normal upward flick or stroke
   - Movement 3: Full / Maximum upward flick or stroke
3. Three WHOLE movements for Scroll DOWN:
   - Movement 1: Slight / Micro downward flick or stroke
   - Movement 2: Medium / Normal downward flick or stroke
   - Movement 3: Full / Maximum downward flick or stroke

Records the complete dynamic motion trajectory:
- Onset acceleration
- Continuous velocity & displacement curve
- Peak deflection and posture snapshot
- Return/recoil deceleration and duration

Saves whole movement trajectories permanently to disk (`gesture_references.json` & `user_profile.json`)
and activates them as the ground truth reference in DenseHandTracker.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("HandScroller.Recorder")

REFERENCES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gesture_references.json")
PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_profile.json")


@dataclass
class HandPoseSnapshot:
    """Snapshot of hand geometry and displacements at one point in time."""
    step_id: str             # 'REST', 'UP_1', 'UP_2', 'UP_3', 'DOWN_1', 'DOWN_2', 'DOWN_3'
    direction: str           # 'REST', 'UP', 'DOWN'
    position_num: int        # 0 for rest, 1..3 for directions
    title: str
    timestamp: float
    scale: float             # Hand scale (wrist to middle knuckle)
    hand_y: float            # Normalized hand center Y
    finger_y: float          # Normalized index tip Y
    middle_y: float          # Normalized middle tip Y
    thumb_y: float           # Normalized thumb tip Y
    disp_index: float        # Displacement from rest in screen Y
    disp_middle: float
    disp_thumb: float
    disp_hand: float
    landmarks_norm: List[Dict[str, float]]  # 21 landmarks normalized relative to wrist/scale
    landmarks_raw: List[Dict[str, float]]   # 21 raw normalized [0..1] landmarks


@dataclass
class MovementTrajectory:
    """Full dynamic trajectory of a recorded whole movement."""
    step_id: str             # 'UP_1', 'UP_2', 'UP_3', 'DOWN_1', 'DOWN_2', 'DOWN_3'
    direction: str           # 'UP' or 'DOWN'
    movement_num: int        # 1, 2, 3
    title: str
    duration: float          # Duration of stroke in seconds (e.g. 0.24s)
    peak_displacement: float # Maximum deflection in screen Y
    peak_velocity: float     # Maximum velocity in screen Y/s
    start_time: float
    end_time: float
    dominant_channel: str    # 'index', 'middle', 'hand'
    trajectory: List[Dict[str, float]] # [{'dt': float, 'disp': float, 'vy': float}]
    onset_snapshot: Dict[str, Any]
    peak_snapshot: Dict[str, Any]
    end_snapshot: Dict[str, Any]


@dataclass
class GestureReferences:
    """Permanent reference dataset saved to disk and loaded by tracking engine."""
    version: int = 2
    recorded_at: float = 0.0
    is_valid: bool = False
    rest_baseline: Optional[Dict[str, Any]] = None
    up_positions: List[Dict[str, Any]] = field(default_factory=list)      # Peak snapshots (3 UP)
    down_positions: List[Dict[str, Any]] = field(default_factory=list)    # Peak snapshots (3 DOWN)
    up_trajectories: List[Dict[str, Any]] = field(default_factory=list)   # Whole movements (3 UP)
    down_trajectories: List[Dict[str, Any]] = field(default_factory=list) # Whole movements (3 DOWN)
    up_threshold: float = 0.012
    down_threshold: float = 0.012
    up_peak_avg: float = 0.035
    down_peak_avg: float = 0.035
    primary_channel: str = 'index'
    up_displacements: List[float] = field(default_factory=list)
    down_displacements: List[float] = field(default_factory=list)


def load_references() -> GestureReferences:
    """Load recorded reference movements from disk if available."""
    # Check gesture_references.json first
    if os.path.exists(REFERENCES_PATH):
        try:
            with open(REFERENCES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                refs = GestureReferences(
                    version=data.get('version', 2),
                    recorded_at=data.get('recorded_at', 0.0),
                    is_valid=data.get('is_valid', False),
                    rest_baseline=data.get('rest_baseline'),
                    up_positions=data.get('up_positions', []),
                    down_positions=data.get('down_positions', []),
                    up_trajectories=data.get('up_trajectories', []),
                    down_trajectories=data.get('down_trajectories', []),
                    up_threshold=float(data.get('up_threshold', 0.012)),
                    down_threshold=float(data.get('down_threshold', 0.012)),
                    up_peak_avg=float(data.get('up_peak_avg', 0.035)),
                    down_peak_avg=float(data.get('down_peak_avg', 0.035)),
                    primary_channel=data.get('primary_channel', 'index'),
                    up_displacements=data.get('up_displacements', []),
                    down_displacements=data.get('down_displacements', []),
                )
                if refs.is_valid and len(refs.up_positions) >= 3 and len(refs.down_positions) >= 3:
                    logger.info("🎯 Loaded personal gesture references (3 UP, 3 DOWN whole movements) from %s", REFERENCES_PATH)
                    return refs
        except Exception as e:
            logger.warning("Could not read references file %s: %s", REFERENCES_PATH, e)

    # Fallback to user_profile.json
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_refs = data.get('gesture_references')
                if raw_refs and isinstance(raw_refs, dict) and raw_refs.get('is_valid'):
                    refs = GestureReferences(
                        version=raw_refs.get('version', 2),
                        recorded_at=raw_refs.get('recorded_at', 0.0),
                        is_valid=True,
                        rest_baseline=raw_refs.get('rest_baseline'),
                        up_positions=raw_refs.get('up_positions', []),
                        down_positions=raw_refs.get('down_positions', []),
                        up_trajectories=raw_refs.get('up_trajectories', []),
                        down_trajectories=raw_refs.get('down_trajectories', []),
                        up_threshold=float(raw_refs.get('up_threshold', 0.012)),
                        down_threshold=float(raw_refs.get('down_threshold', 0.012)),
                        up_peak_avg=float(raw_refs.get('up_peak_avg', 0.035)),
                        down_peak_avg=float(raw_refs.get('down_peak_avg', 0.035)),
                        primary_channel=raw_refs.get('primary_channel', 'index'),
                        up_displacements=raw_refs.get('up_displacements', []),
                        down_displacements=raw_refs.get('down_displacements', []),
                    )
                    return refs
        except Exception as e:
            logger.warning("Could not load references from profile: %s", e)

    return GestureReferences()


def save_references(refs: GestureReferences) -> bool:
    """Save recorded reference movements permanently to disk."""
    try:
        data = asdict(refs)
        # 1. Save to gesture_references.json
        with open(REFERENCES_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info("💾 Saved permanent gesture references to %s", REFERENCES_PATH)

        # 2. Also update user_profile.json so all settings stay in sync
        profile_data = {}
        if os.path.exists(PROFILE_PATH):
            try:
                with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
            except Exception:
                profile_data = {}

        profile_data['calibrated'] = True
        profile_data['calibrated_at'] = refs.recorded_at or time.time()
        base_scale = refs.rest_baseline.get('scale', 0.16) if refs.rest_baseline else 0.16
        base_scale = max(0.015, base_scale)
        scaled_finger = max(0.018, min(0.065, refs.up_threshold / base_scale))
        profile_data['finger_distance'] = float(scaled_finger)
        profile_data['hand_distance'] = float(min(0.085, scaled_finger * 1.4))
        profile_data['gesture_references'] = data
        profile_data['has_recorded_references'] = True

        with open(PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2)
        logger.info("💾 Synchronized references into %s", PROFILE_PATH)
        return True
    except Exception as e:
        logger.error("Failed to save gesture references: %s", e)
        return False


def is_space_or_enter_held() -> bool:
    """Check if SPACE (0x20) or ENTER (0x0D) is currently physically held down on Windows."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        # VK_SPACE = 0x20, VK_RETURN = 0x0D. High bit set if pressed.
        space_down = bool(user32.GetAsyncKeyState(0x20) & 0x8000)
        enter_down = bool(user32.GetAsyncKeyState(0x0D) & 0x8000)
        return space_down or enter_down
    except Exception:
        return False


class GestureRecorder:
    """
    Interactive Whole-Movement Reference Recorder:
    Push-To-Record: Hold SPACE to record the exact start and finish of each movement.
    - Neutral / Resting baseline
    - 3 Whole Movements for Scroll UP (Slight, Medium, Full)
    - 3 Whole Movements for Scroll DOWN (Slight, Medium, Full)
    """

    STEPS = [
        {
            'id': 'REST',
            'direction': 'REST',
            'pos_num': 0,
            'title': 'Resting Neutral Baseline',
            'subtitle': 'Rest hand comfortably on desk (calibrating baseline)',
            'hint': 'PRESS & HOLD SPACE for ~1s while resting hand -> Release when 100%',
        },
        {
            'id': 'UP_1',
            'direction': 'UP',
            'pos_num': 1,
            'title': 'Scroll UP - Movement 1 of 3 (Slight / Micro)',
            'subtitle': 'Perform a SLIGHT / GENTLE upward flick or stroke',
            'hint': 'HOLD SPACE -> Move UP gently -> RELEASE SPACE when done',
        },
        {
            'id': 'UP_2',
            'direction': 'UP',
            'pos_num': 2,
            'title': 'Scroll UP - Movement 2 of 3 (Medium / Normal)',
            'subtitle': 'Perform your NORMAL everyday upward flick or stroke',
            'hint': 'HOLD SPACE -> Move UP normally -> RELEASE SPACE when done',
        },
        {
            'id': 'UP_3',
            'direction': 'UP',
            'pos_num': 3,
            'title': 'Scroll UP - Movement 3 of 3 (Full / Strong)',
            'subtitle': 'Perform your FULL / DECISIVE upward flick or stroke',
            'hint': 'HOLD SPACE -> Swipe UP fully -> RELEASE SPACE when done',
        },
        {
            'id': 'DOWN_1',
            'direction': 'DOWN',
            'pos_num': 1,
            'title': 'Scroll DOWN - Movement 1 of 3 (Slight / Micro)',
            'subtitle': 'Perform a SLIGHT / GENTLE downward flick or stroke',
            'hint': 'HOLD SPACE -> Move DOWN gently -> RELEASE SPACE when done',
        },
        {
            'id': 'DOWN_2',
            'direction': 'DOWN',
            'pos_num': 2,
            'title': 'Scroll DOWN - Movement 2 of 3 (Medium / Normal)',
            'subtitle': 'Perform your NORMAL everyday downward flick or stroke',
            'hint': 'HOLD SPACE -> Move DOWN normally -> RELEASE SPACE when done',
        },
        {
            'id': 'DOWN_3',
            'direction': 'DOWN',
            'pos_num': 3,
            'title': 'Scroll DOWN - Movement 3 of 3 (Full / Strong)',
            'subtitle': 'Perform your FULL / DECISIVE downward flick or stroke',
            'hint': 'HOLD SPACE -> Swipe DOWN fully -> RELEASE SPACE when done',
        },
    ]

    REST_HOLD_SECONDS = 0.85        # Hold rest steady for 0.85s to calibrate neutral
    TRANSITION_DELAY_SECONDS = 0.70 # Brief pause between steps showing green check

    def __init__(self, on_complete_callback=None):
        self.on_complete_callback = on_complete_callback
        self.active = False
        self.current_step_idx = 0
        self.step_start_time = 0.0
        self.hold_start_time = None
        self.hold_progress = 0.0
        self.in_transition = False
        self.transition_start_time = 0.0

        # Hand tracking baseline
        self.rest_baseline: Optional[HandPoseSnapshot] = None
        self.recorded_up: List[HandPoseSnapshot] = []
        self.recorded_down: List[HandPoseSnapshot] = []
        self.recorded_up_trajectories: List[Dict[str, Any]] = []
        self.recorded_down_trajectories: List[Dict[str, Any]] = []

        # Real-time tracking state
        self.prev_finger_y: Optional[float] = None
        self.prev_hand_y: Optional[float] = None
        self.prev_time: float = 0.0
        self.stability_samples: List[float] = []

        # Whole-movement active stroke tracking
        self.is_recording_stroke: bool = False
        self.stroke_start_time: float = 0.0
        self.stroke_samples: List[Dict[str, float]] = []
        self.stroke_onset_snapshot: Optional[HandPoseSnapshot] = None
        self.stroke_peak_snapshot: Optional[HandPoseSnapshot] = None
        self.stroke_peak_disp: float = 0.0
        self.stroke_peak_vel: float = 0.0
        self.stroke_disp: float = 0.0
        self.last_completed_movement: Optional[Dict[str, Any]] = None

        # Current frame capture candidate
        self.last_valid_snapshot: Optional[HandPoseSnapshot] = None
        self.status_message = ""
        self.saved_references: Optional[GestureReferences] = load_references()

        # Keyboard hold push-to-record state
        self.key_held: bool = False
        self.prev_key_held: bool = False
        self.key_down_time: Optional[float] = None
        self.waiting_for_release: bool = False

    def check_physical_key_held(self) -> bool:
        """Return True if SPACE or ENTER is currently physically held down."""
        return is_space_or_enter_held()

    def start(self):
        """Start or restart the 6-movement reference recording process."""
        self.active = True
        self.current_step_idx = 0
        self.step_start_time = time.monotonic()
        self.hold_start_time = None
        self.hold_progress = 0.0
        self.in_transition = False
        self.recorded_up = []
        self.recorded_down = []
        self.recorded_up_trajectories = []
        self.recorded_down_trajectories = []
        self.rest_baseline = None
        self.prev_finger_y = None
        self.prev_hand_y = None
        self.prev_time = 0.0
        self.stability_samples = []
        self.is_recording_stroke = False
        self.stroke_samples = []
        self.stroke_peak_disp = 0.0
        self.stroke_peak_vel = 0.0
        self.stroke_disp = 0.0
        self.last_completed_movement = None
        self.last_valid_snapshot = None
        self.key_held = False
        self.prev_key_held = False
        self.key_down_time = None
        self.waiting_for_release = False
        self.status_message = "Step 1 of 7: Rest hand on desk -> PRESS & HOLD SPACE to record baseline"
        logger.info("🎬 Started Whole Movement Reference Recorder: 3 UP movements, 3 DOWN movements + Resting Baseline")

    def cancel(self):
        """Cancel recording mode."""
        self.active = False
        self.is_recording_stroke = False
        self.key_held = False
        self.prev_key_held = False
        self.waiting_for_release = False
        logger.info("Reference recording cancelled")

    def is_active(self) -> bool:
        return self.active

    def _extract_snapshot(self, landmarks, now: float) -> Optional[HandPoseSnapshot]:
        """Convert MediaPipe landmarks into a normalized HandPoseSnapshot."""
        if not landmarks:
            return None

        # Handle nested landmarks list [[p0..p20]] or [p0..p20]
        if isinstance(landmarks, (list, tuple)) and len(landmarks) > 0 and isinstance(landmarks[0], (list, tuple)):
            landmarks = landmarks[0]

        if len(landmarks) < 21:
            return None

        wrist = landmarks[0]
        knuckle = landmarks[9]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]

        scale = max(0.015, math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y))
        hand_y = (wrist.y + knuckle.y) / 2.0
        finger_y = index_tip.y
        middle_y = middle_tip.y
        thumb_y = thumb_tip.y

        # Compute displacements from rest baseline in screen Y
        if self.rest_baseline:
            base = self.rest_baseline
            disp_hand = hand_y - base.hand_y
            disp_index = finger_y - base.finger_y
            disp_middle = middle_y - base.middle_y
            disp_thumb = thumb_y - base.thumb_y
        else:
            disp_hand = 0.0
            disp_index = 0.0
            disp_middle = 0.0
            disp_thumb = 0.0

        # Normalized coordinates relative to wrist and hand scale
        norm_lms = []
        raw_lms = []
        for lm in landmarks:
            raw_lms.append({'x': float(lm.x), 'y': float(lm.y), 'z': float(getattr(lm, 'z', 0.0))})
            norm_lms.append({
                'x': float((lm.x - wrist.x) / scale),
                'y': float((lm.y - wrist.y) / scale),
                'z': float(getattr(lm, 'z', 0.0) / scale)
            })

        step_info = self.STEPS[self.current_step_idx]
        return HandPoseSnapshot(
            step_id=step_info['id'],
            direction=step_info['direction'],
            position_num=step_info['pos_num'],
            title=step_info['title'],
            timestamp=now,
            scale=scale,
            hand_y=hand_y,
            finger_y=finger_y,
            middle_y=middle_y,
            thumb_y=thumb_y,
            disp_index=disp_index,
            disp_middle=disp_middle,
            disp_thumb=disp_thumb,
            disp_hand=disp_hand,
            landmarks_norm=norm_lms,
            landmarks_raw=raw_lms,
        )

    def manual_capture(self, now: Optional[float] = None) -> bool:
        """Lock in position immediately when user presses SPACE / ENTER."""
        if not self.active or self.in_transition or not self.last_valid_snapshot:
            return False
        capture_time = now if now is not None else (
            self.last_valid_snapshot.timestamp if self.last_valid_snapshot and getattr(self.last_valid_snapshot, 'timestamp', 0) > 0
            else time.monotonic()
        )
        logger.info("Manual capture triggered for step %s", self.STEPS[self.current_step_idx]['id'])

        step_info = self.STEPS[self.current_step_idx]
        if step_info['id'] == 'REST':
            self.rest_baseline = self.last_valid_snapshot
            self.in_transition = True
            self.transition_start_time = capture_time
            self.status_message = "✓ Resting Baseline Saved!"
        else:
            # Complete stroke with whatever motion was observed or current snapshot
            peak_snap = self.stroke_peak_snapshot or self.last_valid_snapshot
            peak_d = max(0.008, self.stroke_peak_disp or abs(peak_snap.disp_index) or 0.015)
            self._finalize_stroke(peak_snap, peak_d, max(0.25, self.stroke_peak_vel or 0.50), capture_time)
        return True

    def _finalize_stroke(self, peak_snapshot: HandPoseSnapshot, peak_disp: float, peak_vel: float, now: float):
        """Complete the current whole movement stroke and save trajectory."""
        step_info = self.STEPS[self.current_step_idx]
        direction = step_info['direction']
        duration = max(0.10, now - self.stroke_start_time) if self.stroke_start_time > 0 else 0.22

        trajectory_data = {
            'step_id': step_info['id'],
            'direction': direction,
            'movement_num': step_info['pos_num'],
            'title': step_info['title'],
            'duration': duration,
            'peak_displacement': float(peak_disp),
            'peak_velocity': float(peak_vel),
            'start_time': self.stroke_start_time,
            'end_time': now,
            'dominant_channel': 'index' if abs(peak_snapshot.disp_index) >= abs(peak_snapshot.disp_hand) * 0.8 else 'hand',
            'trajectory': list(self.stroke_samples) if self.stroke_samples else [{'dt': 0.0, 'disp': peak_disp, 'vy': peak_vel}],
            'onset_snapshot': asdict(self.stroke_onset_snapshot) if self.stroke_onset_snapshot else asdict(peak_snapshot),
            'peak_snapshot': asdict(peak_snapshot),
            'end_snapshot': asdict(peak_snapshot),
        }

        if direction == 'UP':
            self.recorded_up_trajectories.append(trajectory_data)
            self.recorded_up.append(peak_snapshot)
            logger.info("✅ Recorded whole UP movement %d/3: peak=%.4f (%.1fpx), vel=%.3f, dur=%.2fs",
                        step_info['pos_num'], peak_disp, peak_disp * 1000, peak_vel, duration)
        elif direction == 'DOWN':
            self.recorded_down_trajectories.append(trajectory_data)
            self.recorded_down.append(peak_snapshot)
            logger.info("✅ Recorded whole DOWN movement %d/3: peak=%.4f (%.1fpx), vel=%.3f, dur=%.2fs",
                        step_info['pos_num'], peak_disp, peak_disp * 1000, peak_vel, duration)

        self.last_completed_movement = trajectory_data
        self.is_recording_stroke = False
        self.stroke_samples = []
        self.in_transition = True
        self.transition_start_time = now
        self.waiting_for_release = True
        self.status_message = f"[SAVED] {step_info['title']} ({peak_disp*1000:.0f}px travel)"

    def update(self, landmarks, frame_rgb=None, now=None, key_held: Optional[bool] = None) -> Dict[str, Any]:
        """
        Feed camera frame / landmarks into the whole-movement recorder.
        Pure Push-To-Record Workflow:
        - The record DOES NOT START until you press and hold SPACE.
        - The record DOES NOT END until you release (lift hand from) SPACE.
        - Upon release, that move is saved as a reference for future use.
        - Next move does not start until you release SPACE first, then press again!
        """
        if not self.active:
            return {'active': False}

        now = time.monotonic() if now is None else now
        dt = max(0.005, min(0.15, now - self.prev_time)) if self.prev_time > 0 else 0.033

        # Check physical key hold state if not passed explicitly
        if key_held is None:
            key_held = self.check_physical_key_held()

        is_key_down_edge = (key_held and not self.prev_key_held)
        is_key_up_edge = (not key_held and self.prev_key_held)
        self.key_held = key_held

        # ─── 1. POST-CAPTURE TRANSITION DELAY ─────────────────────────
        if self.in_transition:
            if not key_held:
                self.waiting_for_release = False

            if now - self.transition_start_time >= self.TRANSITION_DELAY_SECONDS:
                self.in_transition = False
                self.current_step_idx += 1
                self.hold_start_time = None
                self.hold_progress = 0.0
                self.is_recording_stroke = False
                self.stroke_samples = []
                self.stroke_peak_disp = 0.0
                self.stroke_peak_vel = 0.0

                if self.current_step_idx >= len(self.STEPS):
                    # All 6 whole movements + rest completed! Save to disk!
                    self._finalize_recording(now)
                    self.prev_key_held = key_held
                    return self.get_hud_info()

                # If the key is still held when transition ends, require release
                if key_held:
                    self.waiting_for_release = True
                else:
                    self.waiting_for_release = False

            self.prev_key_held = key_held
            return self.get_hud_info()

        if self.current_step_idx >= len(self.STEPS):
            self.prev_key_held = key_held
            return self.get_hud_info()

        # ─── 2. WAIT FOR KEY RELEASE BEFORE NEXT MOVE ─────────────────
        if self.waiting_for_release:
            if not key_held:
                self.waiting_for_release = False
            else:
                next_step = self.STEPS[self.current_step_idx]
                self.status_message = f"Release SPACE to prepare for {next_step['title']}"
                self.prev_key_held = key_held
                return self.get_hud_info()

        step_info = self.STEPS[self.current_step_idx]
        snapshot = self._extract_snapshot(landmarks, now)
        self.last_valid_snapshot = snapshot

        if not snapshot:
            self.hold_start_time = None
            self.hold_progress = 0.0
            self.is_recording_stroke = False
            self.status_message = "Keep your hand visible in camera view"
            self.prev_key_held = key_held
            return self.get_hud_info()

        # Compute instant vertical velocities
        if self.prev_finger_y is not None:
            finger_vy = (snapshot.finger_y - self.prev_finger_y) / dt
            movement = abs(snapshot.finger_y - self.prev_finger_y) / snapshot.scale
            self.stability_samples.append(movement)
            if len(self.stability_samples) > 10:
                self.stability_samples.pop(0)
            avg_movement = sum(self.stability_samples) / len(self.stability_samples)
        else:
            finger_vy = 0.0
            avg_movement = 0.0

        self.prev_finger_y = snapshot.finger_y
        self.prev_hand_y = snapshot.hand_y
        self.prev_time = now

        # ─── 3. PHASE 1: RESTING NEUTRAL BASELINE (Step 0) ───────────
        if step_info['id'] == 'REST':
            if key_held:
                if self.hold_start_time is None:
                    self.hold_start_time = now
                hold_elapsed = now - self.hold_start_time
                self.hold_progress = min(1.0, hold_elapsed / self.REST_HOLD_SECONDS)
                pct = int(self.hold_progress * 100)
                self.status_message = f"Calibrating resting baseline: {pct}% (Keep holding SPACE)"
                if self.hold_progress >= 1.0:
                    self.rest_baseline = snapshot
                    self.in_transition = True
                    self.transition_start_time = now
                    self.waiting_for_release = True
                    self.status_message = "[OK] Resting Baseline Calibrated! Now release SPACE"
                    logger.info("✅ Resting Baseline Calibrated via SPACE hold: scale=%.3f, hand_y=%.3f",
                                snapshot.scale, snapshot.hand_y)
            elif is_key_up_edge and self.hold_progress >= 0.40:
                self.rest_baseline = snapshot
                self.in_transition = True
                self.transition_start_time = now
                self.waiting_for_release = False
                self.status_message = "[OK] Resting Baseline Calibrated!"
            else:
                self.hold_start_time = None
                self.hold_progress = 0.0
                self.status_message = "Rest hand on desk -> PRESS & HOLD SPACE to record baseline"

            if is_key_down_edge:
                self.key_down_time = now
            self.prev_key_held = key_held
            return self.get_hud_info()

        # ─── 4. PHASE 2 & 3: WHOLE MOVEMENTS (Steps 1..6) ─────────────
        direction = step_info['direction']

        # Displacement in intended direction (positive = moving in target direction)
        if direction == 'UP' and self.rest_baseline:
            disp = self.rest_baseline.finger_y - snapshot.finger_y
            intended_vy = -finger_vy  # positive when moving up
        elif direction == 'DOWN' and self.rest_baseline:
            disp = snapshot.finger_y - self.rest_baseline.finger_y
            intended_vy = finger_vy   # positive when moving down
        else:
            disp = 0.0
            intended_vy = 0.0

        self.stroke_disp = disp

        # EVENT A: User pressed key down -> Start Stroke Recording
        if is_key_down_edge:
            self.key_down_time = now
            self.is_recording_stroke = True
            self.stroke_start_time = now
            self.stroke_onset_snapshot = snapshot
            self.stroke_peak_snapshot = snapshot
            self.stroke_peak_disp = max(0.0, disp)
            self.stroke_peak_vel = abs(intended_vy)
            self.stroke_samples = [{'dt': 0.0, 'disp': disp, 'vy': intended_vy, 'finger_y': snapshot.finger_y}]
            logger.info("🔴 Started recording %s stroke (SPACE pressed down)", step_info['id'])

        # EVENT B: User released key (lifted hand from button) -> Finish & Save Whole Movement
        if is_key_up_edge and self.is_recording_stroke:
            stroke_elapsed = max(0.05, now - self.stroke_start_time)
            peak_snap = self.stroke_peak_snapshot or snapshot
            peak_d = max(0.005, self.stroke_peak_disp or abs(snapshot.disp_index) or 0.015)
            peak_v = max(0.20, self.stroke_peak_vel or 0.50)
            self._finalize_stroke(peak_snap, peak_d, peak_v, now)
            self.waiting_for_release = False
            self.prev_key_held = key_held
            return self.get_hud_info()

        # EVENT C: Key is held down -> Stream Live Movement
        if key_held and self.is_recording_stroke:
            stroke_elapsed = now - self.stroke_start_time
            if disp > self.stroke_peak_disp:
                self.stroke_peak_disp = disp
                self.stroke_peak_snapshot = snapshot
            self.stroke_peak_vel = max(self.stroke_peak_vel, abs(intended_vy))
            self.stroke_samples.append({
                'dt': stroke_elapsed,
                'disp': disp,
                'vy': intended_vy,
                'finger_y': snapshot.finger_y
            })
            self.status_message = f"[REC] {direction}: {disp*1000:.0f}px (Peak: {self.stroke_peak_disp*1000:.0f}px) - RELEASE SPACE TO FINISH"
            self.prev_key_held = key_held
            return self.get_hud_info()

        # EVENT D: Key NOT held and not recording -> Waiting for user to press SPACE
        if not key_held and not self.is_recording_stroke:
            self.status_message = f"PRESS & HOLD SPACE -> Perform {direction} movement -> RELEASE SPACE to finish"

        self.prev_key_held = key_held
        return self.get_hud_info()

    def _finalize_recording(self, now: float):
        """
        Process recorded whole movements, compute personal thresholds,
        save to disk permanently, and trigger on_complete callback.
        """
        self.active = False
        logger.info("🎉 Finalizing 6 whole movement references...")

        # Extract peak displacements
        up_disps = [float(m['peak_displacement']) for m in self.recorded_up_trajectories]
        down_disps = [float(m['peak_displacement']) for m in self.recorded_down_trajectories]

        up_disps.sort()
        down_disps.sort()

        # Trigger at 65% of the user's smallest recorded whole movement (Move 1 - Slight)
        up_thresh = max(0.003, up_disps[0] * 0.65) if up_disps else 0.012
        down_thresh = max(0.003, down_disps[0] * 0.65) if down_disps else 0.012

        up_peak_avg = float(np.mean(up_disps)) if up_disps else 0.035
        down_peak_avg = float(np.mean(down_disps)) if down_disps else 0.035

        # Check primary active channel from recorded movements
        channels = [m.get('dominant_channel', 'index') for m in self.recorded_up_trajectories + self.recorded_down_trajectories]
        primary_ch = max(set(channels), key=channels.count) if channels else 'index'

        refs = GestureReferences(
            version=2,
            recorded_at=now,
            is_valid=True,
            rest_baseline=asdict(self.rest_baseline) if self.rest_baseline else None,
            up_positions=[asdict(p) for p in self.recorded_up],
            down_positions=[asdict(p) for p in self.recorded_down],
            up_trajectories=self.recorded_up_trajectories,
            down_trajectories=self.recorded_down_trajectories,
            up_threshold=float(up_thresh),
            down_threshold=float(down_thresh),
            up_peak_avg=up_peak_avg,
            down_peak_avg=down_peak_avg,
            primary_channel=primary_ch,
            up_displacements=up_disps,
            down_displacements=down_disps,
        )

        self.saved_references = refs
        save_references(refs)

        logger.info("🎉 SUCCESS! Saved 6 personal whole-movement trajectories:")
        logger.info("   UP Movements: peaks=%s (Threshold: %.4f, Avg Peak: %.4f)",
                    [f"{d:.3f}" for d in up_disps], up_thresh, up_peak_avg)
        logger.info("   DOWN Movements: peaks=%s (Threshold: %.4f, Avg Peak: %.4f)",
                    [f"{d:.3f}" for d in down_disps], down_thresh, down_peak_avg)
        logger.info("   Primary Channel: %s", primary_ch)

        if self.on_complete_callback:
            self.on_complete_callback(refs)

    def get_hud_info(self) -> Dict[str, Any]:
        """Provides status data for rendering in UIOverlay."""
        step_idx = min(len(self.STEPS) - 1, self.current_step_idx)
        step_info = self.STEPS[step_idx]

        return {
            'active': self.active,
            'is_complete': not self.active and self.saved_references and self.saved_references.is_valid,
            'step_index': self.current_step_idx + 1,
            'total_steps': len(self.STEPS),
            'step_id': step_info['id'],
            'direction': step_info['direction'],
            'pos_num': step_info['pos_num'],
            'title': step_info['title'],
            'subtitle': step_info['subtitle'],
            'hint': step_info['hint'],
            'status_message': self.status_message,
            'hold_progress': self.hold_progress,
            'key_held': self.key_held,
            'waiting_for_release': self.waiting_for_release,
            'is_recording_stroke': self.is_recording_stroke,
            'stroke_disp': self.stroke_disp,
            'stroke_peak_disp': self.stroke_peak_disp,
            'in_transition': self.in_transition,
            'recorded_up_count': len(self.recorded_up_trajectories),
            'recorded_down_count': len(self.recorded_down_trajectories),
            'has_rest': self.rest_baseline is not None,
            'last_movement': self.last_completed_movement,
        }
