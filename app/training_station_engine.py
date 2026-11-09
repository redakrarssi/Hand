"""
Training Station Engine Module - Multi-Mode Hand Movement Capture & Telemetry.

Manages recording sessions for:
1. Guided Kinematics (Resting baseline + 3 UP flicks + 3 DOWN flicks + optional touch)
2. Freeform Session (Natural live scrolling / browsing recording with auto-segmentation)
3. Repetition Drill (5x UP flicks, 5x DOWN flicks for variance and repeatability)

Logs high-resolution multi-landmark trajectories, instantaneous velocities,
camera noise floor samples, and outputs structured datasets to `recordings/`.
"""

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

from app.kinematics_analyzer import KinematicsAnalyzer, UserKinematicsProfile

logger = logging.getLogger("HandScroller.TrainingStation")

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")
REFERENCES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gesture_references.json")
PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_profile.json")


def is_space_or_enter_held() -> bool:
    """Check physical status of SPACE (0x20) or ENTER (0x0D) on Windows."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        space = bool(user32.GetAsyncKeyState(0x20) & 0x8000)
        enter = bool(user32.GetAsyncKeyState(0x0D) & 0x8000)
        return space or enter
    except Exception:
        return False


@dataclass
class FrameTelemetry:
    """Frame-by-frame raw hand tracking data."""
    t: float
    dt: float
    wrist_y: float
    index_y: float
    middle_y: float
    thumb_y: float
    hand_scale: float
    disp_index: float
    disp_middle: float
    disp_wrist: float
    vy_index: float
    vy_middle: float
    vy_wrist: float
    key_held: bool
    landmarks_norm: List[Dict[str, float]] = field(default_factory=list)


class TrainingStationEngine:
    """
    Core engine that coordinates camera frames, user inputs, motion detection,
    and dataset saving.
    """

    MODE_GUIDED = "GUIDED"       # Step-by-step calibration
    MODE_FREEFORM = "FREEFORM"   # Continuous live natural movement
    MODE_DRILL = "DRILL"         # 5 UP + 5 DOWN repetition test

    GUIDED_STEPS = [
        {
            'id': 'REST',
            'direction': 'REST',
            'title': 'Resting Neutral Baseline',
            'desc': 'Rest hand comfortably on desk (calibrating camera noise)',
            'hint': 'HOLD SPACE for ~1s while resting hand steady -> Release when 100%',
        },
        {
            'id': 'UP_1',
            'direction': 'UP',
            'title': 'Scroll UP - Stroke 1/3 (Slight / Micro)',
            'desc': 'Perform a GENTLE / MICRO upward flick with your index finger',
            'hint': 'HOLD SPACE -> Flick UP gently -> RELEASE SPACE when done',
        },
        {
            'id': 'UP_2',
            'direction': 'UP',
            'title': 'Scroll UP - Stroke 2/3 (Normal / Everyday)',
            'desc': 'Perform your NORMAL everyday upward flick or stroke',
            'hint': 'HOLD SPACE -> Flick UP normally -> RELEASE SPACE when done',
        },
        {
            'id': 'UP_3',
            'direction': 'UP',
            'title': 'Scroll UP - Stroke 3/3 (Full / Decisive)',
            'desc': 'Perform your FULL / DECISIVE upward flick or stroke',
            'hint': 'HOLD SPACE -> Flick UP decisively -> RELEASE SPACE when done',
        },
        {
            'id': 'DOWN_1',
            'direction': 'DOWN',
            'title': 'Scroll DOWN - Stroke 1/3 (Slight / Micro)',
            'desc': 'Perform a GENTLE / MICRO downward flick with your index finger',
            'hint': 'HOLD SPACE -> Flick DOWN gently -> RELEASE SPACE when done',
        },
        {
            'id': 'DOWN_2',
            'direction': 'DOWN',
            'title': 'Scroll DOWN - Stroke 2/3 (Normal / Everyday)',
            'desc': 'Perform your NORMAL everyday downward flick or stroke',
            'hint': 'HOLD SPACE -> Flick DOWN normally -> RELEASE SPACE when done',
        },
        {
            'id': 'DOWN_3',
            'direction': 'DOWN',
            'title': 'Scroll DOWN - Stroke 3/3 (Full / Decisive)',
            'desc': 'Perform your FULL / DECISIVE downward flick or stroke',
            'hint': 'HOLD SPACE -> Flick DOWN decisively -> RELEASE SPACE when done',
        },
    ]

    def __init__(self, on_session_completed: Optional[Callable[[UserKinematicsProfile], None]] = None):
        self.on_session_completed = on_session_completed
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        self.mode = self.MODE_GUIDED
        self.active = False
        self.session_id = ""
        self.session_start_time = 0.0

        # Step management for GUIDED mode
        self.step_idx = 0
        self.step_start_time = 0.0
        self.hold_progress = 0.0
        self.hold_start_time = None
        self.in_transition = False
        self.transition_start = 0.0
        self.waiting_for_release = False

        # Drill mode management
        self.drill_target_direction = "UP"
        self.drill_count = 0
        self.drill_target_total = 5

        # Resting baseline
        self.baseline_wrist_y = None
        self.baseline_index_y = None
        self.baseline_middle_y = None
        self.baseline_scale = 0.16
        self.rest_samples: List[Dict[str, float]] = []

        # Real-time state
        self.prev_time = 0.0
        self.prev_index_y = None
        self.prev_middle_y = None
        self.prev_wrist_y = None
        self.key_held = False
        self.prev_key_held = False

        # Stroke recording
        self.is_recording_stroke = False
        self.stroke_start_time = 0.0
        self.stroke_samples: List[Dict[str, float]] = []
        self.stroke_peak_disp = 0.0
        self.stroke_peak_vel = 0.0
        self.stroke_direction = "UP"
        self.stroke_title = ""

        # Captured session data
        self.completed_movements: List[Dict[str, Any]] = []
        self.all_frame_telemetry: List[Dict[str, Any]] = []

        # Oscilloscope ring buffer for HUD waveform (dt, disp, vy)
        self.oscilloscope_buffer = deque(maxlen=120)

        # Last saved report & profile
        self.latest_profile: Optional[UserKinematicsProfile] = None
        self.status_message = "Ready. Press 1 for Guided, 2 for Freeform, 3 for Drill."

    def start_session(self, mode: str = MODE_GUIDED):
        """Start a new recording session in specified mode."""
        self.mode = mode
        self.active = True
        self.session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_start_time = time.monotonic()
        self.step_idx = 0
        self.step_start_time = self.session_start_time
        self.hold_progress = 0.0
        self.hold_start_time = None
        self.in_transition = False
        self.waiting_for_release = False
        self.is_recording_stroke = False
        self.completed_movements = []
        self.all_frame_telemetry = []
        self.rest_samples = []
        self.drill_count = 0
        self.drill_target_direction = "UP"

        if mode == self.MODE_GUIDED:
            self.status_message = "Step 1/7: Rest hand steady -> HOLD SPACE to calibrate baseline"
        elif mode == self.MODE_FREEFORM:
            self.status_message = "FREEFORM MODE: Perform natural scrolls/flicks. SPACE=Tag/Record, Q=Finish"
        elif mode == self.MODE_DRILL:
            self.status_message = "DRILL MODE: 5 UP flicks, then 5 DOWN flicks. Hold SPACE during each flick"

        logger.info("🚀 Started Training Station Session [%s] in mode %s", self.session_id, mode)

    def cancel_session(self):
        """Cancel current session without saving."""
        self.active = False
        self.is_recording_stroke = False
        self.status_message = "Session cancelled."
        logger.info("❌ Recording session cancelled.")

    def update(self, landmarks_list, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a single camera frame with hand landmarks.
        Returns HUD telemetry dictionary for visualization.
        """
        if now is None:
            now = time.monotonic()

        dt = max(0.001, min(0.1, now - self.prev_time)) if self.prev_time > 0 else 0.033
        key_held = is_space_or_enter_held()
        is_key_down_edge = key_held and not self.prev_key_held
        is_key_up_edge = not key_held and self.prev_key_held

        # Check if landmarks are present
        has_hand = False
        landmarks = None
        if landmarks_list:
            if isinstance(landmarks_list, (list, tuple)) and len(landmarks_list) > 0:
                landmarks = landmarks_list[0] if isinstance(landmarks_list[0], (list, tuple)) else landmarks_list
                if len(landmarks) >= 21:
                    has_hand = True

        if not has_hand or not self.active:
            self.prev_key_held = key_held
            self.prev_time = now
            return self.get_hud_info(has_hand=False)

        # Extract anatomical coordinates
        wrist = landmarks[0]
        knuckle = landmarks[9]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]

        scale = max(0.015, math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y))
        wrist_y = float(wrist.y)
        index_y = float(index_tip.y)
        middle_y = float(middle_tip.y)
        thumb_y = float(thumb_tip.y)

        # Instantaneous velocities
        vy_index = (index_y - self.prev_index_y) / dt if self.prev_index_y is not None else 0.0
        vy_middle = (middle_y - self.prev_middle_y) / dt if self.prev_middle_y is not None else 0.0
        vy_wrist = (wrist_y - self.prev_wrist_y) / dt if self.prev_wrist_y is not None else 0.0

        # Displacements from baseline
        if self.baseline_index_y is not None:
            disp_index = self.baseline_index_y - index_y   # Positive when moving UP
            disp_middle = self.baseline_middle_y - middle_y
            disp_wrist = self.baseline_wrist_y - wrist_y
        else:
            disp_index = 0.0
            disp_middle = 0.0
            disp_wrist = 0.0

        # Feed Oscilloscope
        self.oscilloscope_buffer.append({
            't': now,
            'disp': disp_index,
            'vy': vy_index,
            'is_moving': abs(disp_index) > 0.012 or abs(vy_index) > 0.15
        })

        # Save frame telemetry
        lms_norm = [{'x': round(float(p.x), 4), 'y': round(float(p.y), 4), 'z': round(float(getattr(p, 'z', 0.0)), 4)} for p in landmarks[:21]]
        frame_tel = FrameTelemetry(
            t=round(now - self.session_start_time, 4),
            dt=round(dt, 4),
            wrist_y=round(wrist_y, 4),
            index_y=round(index_y, 4),
            middle_y=round(middle_y, 4),
            thumb_y=round(thumb_y, 4),
            hand_scale=round(scale, 4),
            disp_index=round(disp_index, 5),
            disp_middle=round(disp_middle, 5),
            disp_wrist=round(disp_wrist, 5),
            vy_index=round(vy_index, 4),
            vy_middle=round(vy_middle, 4),
            vy_wrist=round(vy_wrist, 4),
            key_held=key_held,
            landmarks_norm=lms_norm
        )
        self.all_frame_telemetry.append(asdict(frame_tel))

        # ─── ROUTE BY MODE ──────────────────────────────────────────
        if self.mode == self.MODE_GUIDED:
            self._update_guided_mode(now, disp_index, disp_wrist, vy_index, scale, index_y, middle_y, wrist_y,
                                     key_held, is_key_down_edge, is_key_up_edge)
        elif self.mode == self.MODE_FREEFORM:
            self._update_freeform_mode(now, disp_index, disp_wrist, vy_index, key_held, is_key_down_edge, is_key_up_edge)
        elif self.mode == self.MODE_DRILL:
            self._update_drill_mode(now, disp_index, disp_wrist, vy_index, key_held, is_key_down_edge, is_key_up_edge)

        self.prev_time = now
        self.prev_index_y = index_y
        self.prev_middle_y = middle_y
        self.prev_wrist_y = wrist_y
        self.prev_key_held = key_held

        return self.get_hud_info(has_hand=True)

    def _update_guided_mode(self, now, disp_index, disp_wrist, vy_index, scale, index_y, middle_y, wrist_y,
                            key_held, is_key_down_edge, is_key_up_edge):
        """Update step-by-step guided calibration."""
        if self.in_transition:
            if now - self.transition_start >= 0.70:
                self.in_transition = False
                self.step_idx += 1
                if self.step_idx >= len(self.GUIDED_STEPS):
                    self._finalize_and_save_session()
                    return
                step = self.GUIDED_STEPS[self.step_idx]
                self.status_message = f"Step {self.step_idx+1}/7: {step['title']} - {step['hint']}"
            return

        step = self.GUIDED_STEPS[self.step_idx]

        # STEP 0: RESTING BASELINE
        if step['id'] == 'REST':
            if key_held:
                if self.hold_start_time is None:
                    self.hold_start_time = now
                elapsed = now - self.hold_start_time
                self.hold_progress = min(1.0, elapsed / 0.85)
                # Sample noise
                jitter = abs(vy_index) * 0.033
                self.rest_samples.append({'scale': scale, 'jitter': jitter, 'disp_index': jitter})
                pct = int(self.hold_progress * 100)
                self.status_message = f"Calibrating Noise Floor: {pct}% (Keep holding steady)"
                if self.hold_progress >= 1.0:
                    self.baseline_wrist_y = wrist_y
                    self.baseline_index_y = index_y
                    self.baseline_middle_y = middle_y
                    self.baseline_scale = scale
                    self.in_transition = True
                    self.transition_start = now
                    self.waiting_for_release = True
                    self.status_message = "✅ Resting Baseline Calibrated! Release SPACE"
                    logger.info("✅ Rest baseline set: index_y=%.3f, scale=%.3f, samples=%d",
                                index_y, scale, len(self.rest_samples))
            else:
                self.hold_start_time = None
                self.hold_progress = 0.0
                self.status_message = "Rest hand comfortably -> HOLD SPACE for ~1s to calibrate"
            return

        # STEPS 1..6: WHOLE MOVEMENTS (UP_1..3, DOWN_1..3)
        target_dir = step['direction']
        # Compute directional stroke displacement
        effective_disp = disp_index if target_dir == "UP" else -disp_index

        if is_key_down_edge:
            self.is_recording_stroke = True
            self.stroke_start_time = now
            self.stroke_peak_disp = max(0.0, effective_disp)
            self.stroke_peak_vel = abs(vy_index)
            self.stroke_direction = target_dir
            self.stroke_title = step['title']
            self.stroke_samples = [{'dt': 0.0, 'disp': effective_disp, 'vy': -vy_index if target_dir == 'UP' else vy_index}]
            self.status_message = f"🔴 Recording {target_dir} stroke... Flick and RELEASE SPACE when done"

        elif key_held and self.is_recording_stroke:
            stroke_dt = now - self.stroke_start_time
            if effective_disp > self.stroke_peak_disp:
                self.stroke_peak_disp = effective_disp
            self.stroke_peak_vel = max(self.stroke_peak_vel, abs(vy_index))
            self.stroke_samples.append({
                'dt': round(stroke_dt, 4),
                'disp': round(effective_disp, 5),
                'vy': round(-vy_index if target_dir == 'UP' else vy_index, 4)
            })
            self.status_message = f"⚡ {target_dir}: {effective_disp*1000:.0f}px (Peak: {self.stroke_peak_disp*1000:.0f}px) - RELEASE SPACE"

        elif is_key_up_edge and self.is_recording_stroke:
            self.is_recording_stroke = False
            duration = max(0.05, now - self.stroke_start_time)
            dominant = "wrist" if abs(disp_wrist) > abs(disp_index) * 0.75 else "index"
            movement_record = {
                'step_id': step['id'],
                'direction': target_dir,
                'title': step['title'],
                'duration': round(duration, 3),
                'peak_displacement': round(max(0.008, self.stroke_peak_disp), 4),
                'peak_velocity': round(max(0.18, self.stroke_peak_vel), 3),
                'dominant_channel': dominant,
                'channel_displacements': {'index': self.stroke_peak_disp, 'wrist': abs(disp_wrist)},
                'trajectory': self.stroke_samples
            }
            self.completed_movements.append(movement_record)
            self.in_transition = True
            self.transition_start = now
            self.status_message = f"✅ Recorded {step['title']}! (Peak: {movement_record['peak_displacement']*1000:.0f}px)"
            logger.info("Recorded %s: peak=%.4f, vel=%.2f, dur=%.2fs",
                        step['id'], movement_record['peak_displacement'],
                        movement_record['peak_velocity'], duration)

    def _update_freeform_mode(self, now, disp_index, disp_wrist, vy_index, key_held, is_key_down_edge, is_key_up_edge):
        """Freeform natural scrolling mode with auto-tagging or push-to-record."""
        if self.baseline_index_y is None:
            self.baseline_index_y = self.prev_index_y
            self.baseline_middle_y = self.prev_middle_y
            self.baseline_wrist_y = self.prev_wrist_y

        direction = "UP" if disp_index > 0 else "DOWN"
        peak_disp = abs(disp_index)

        if is_key_down_edge:
            self.is_recording_stroke = True
            self.stroke_start_time = now
            self.stroke_peak_disp = peak_disp
            self.stroke_peak_vel = abs(vy_index)
            self.stroke_direction = direction
            self.stroke_samples = [{'dt': 0.0, 'disp': disp_index, 'vy': vy_index}]
            self.status_message = f"🔴 Recording Freeform Gesture ({direction})..."

        elif key_held and self.is_recording_stroke:
            stroke_dt = now - self.stroke_start_time
            if peak_disp > self.stroke_peak_disp:
                self.stroke_peak_disp = peak_disp
                self.stroke_direction = direction
            self.stroke_peak_vel = max(self.stroke_peak_vel, abs(vy_index))
            self.stroke_samples.append({
                'dt': round(stroke_dt, 4),
                'disp': round(disp_index, 5),
                'vy': round(vy_index, 4)
            })

        elif is_key_up_edge and self.is_recording_stroke:
            self.is_recording_stroke = False
            duration = max(0.05, now - self.stroke_start_time)
            movement_record = {
                'step_id': f"FREE_{len(self.completed_movements)+1}",
                'direction': self.stroke_direction,
                'title': f"Natural {self.stroke_direction} Stroke #{len(self.completed_movements)+1}",
                'duration': round(duration, 3),
                'peak_displacement': round(max(0.008, self.stroke_peak_disp), 4),
                'peak_velocity': round(max(0.18, self.stroke_peak_vel), 3),
                'dominant_channel': 'index',
                'channel_displacements': {'index': self.stroke_peak_disp, 'wrist': abs(disp_wrist)},
                'trajectory': self.stroke_samples
            }
            self.completed_movements.append(movement_record)
            self.status_message = f"✅ Logged {self.stroke_direction} stroke #{len(self.completed_movements)}! Press Q to finish."

    def _update_drill_mode(self, now, disp_index, disp_wrist, vy_index, key_held, is_key_down_edge, is_key_up_edge):
        """Repetition drill (5x UP, then 5x DOWN)."""
        target_dir = self.drill_target_direction
        effective_disp = disp_index if target_dir == "UP" else -disp_index

        if is_key_down_edge:
            self.is_recording_stroke = True
            self.stroke_start_time = now
            self.stroke_peak_disp = max(0.0, effective_disp)
            self.stroke_peak_vel = abs(vy_index)
            self.stroke_samples = [{'dt': 0.0, 'disp': effective_disp, 'vy': -vy_index if target_dir == 'UP' else vy_index}]
            self.status_message = f"🔴 Drill {target_dir} ({self.drill_count+1}/5)... Release SPACE to log"

        elif key_held and self.is_recording_stroke:
            stroke_dt = now - self.stroke_start_time
            if effective_disp > self.stroke_peak_disp:
                self.stroke_peak_disp = effective_disp
            self.stroke_peak_vel = max(self.stroke_peak_vel, abs(vy_index))
            self.stroke_samples.append({
                'dt': round(stroke_dt, 4),
                'disp': round(effective_disp, 5),
                'vy': round(-vy_index if target_dir == 'UP' else vy_index, 4)
            })

        elif is_key_up_edge and self.is_recording_stroke:
            self.is_recording_stroke = False
            duration = max(0.05, now - self.stroke_start_time)
            movement_record = {
                'step_id': f"DRILL_{target_dir}_{self.drill_count+1}",
                'direction': target_dir,
                'title': f"{target_dir} Repetition #{self.drill_count+1}",
                'duration': round(duration, 3),
                'peak_displacement': round(max(0.008, self.stroke_peak_disp), 4),
                'peak_velocity': round(max(0.18, self.stroke_peak_vel), 3),
                'dominant_channel': 'index',
                'channel_displacements': {'index': self.stroke_peak_disp, 'wrist': abs(disp_wrist)},
                'trajectory': self.stroke_samples
            }
            self.completed_movements.append(movement_record)
            self.drill_count += 1
            if self.drill_count >= self.drill_target_total:
                if self.drill_target_direction == "UP":
                    self.drill_target_direction = "DOWN"
                    self.drill_count = 0
                    self.status_message = "✅ 5x UP Repetitions Complete! Now performing 5x DOWN Repetitions"
                else:
                    self._finalize_and_save_session()
            else:
                self.status_message = f"✅ Logged {target_dir} ({self.drill_count}/5). Do flick #{self.drill_count+1}!"

    def _finalize_and_save_session(self):
        """Save dataset to disk and run KinematicsAnalyzer."""
        self.active = False
        logger.info("💾 Finalizing training station session [%s]...", self.session_id)

        session_data = {
            'session_id': self.session_id,
            'timestamp': time.time(),
            'mode': self.mode,
            'movements_count': len(self.completed_movements),
            'telemetry_frames_count': len(self.all_frame_telemetry),
            'rest_samples': self.rest_samples,
            'movements': self.completed_movements,
            'raw_telemetry': self.all_frame_telemetry
        }

        # 1. Save timestamped session
        session_file = os.path.join(RECORDINGS_DIR, f"{self.session_id}.json")
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            logger.info("💾 Saved full session to %s", session_file)
        except Exception as e:
            logger.error("Failed to write session file: %s", e)

        # 2. Save latest_session.json copy
        latest_file = os.path.join(RECORDINGS_DIR, "latest_session.json")
        try:
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            logger.info("💾 Updated %s", latest_file)
        except Exception as e:
            logger.error("Failed to update latest session: %s", e)

        # 3. Analyze kinematics & generate recommendations
        profile = KinematicsAnalyzer.analyze_session_data(session_data)
        self.latest_profile = profile

        # 4. Save Kinematics Summary JSON
        summary_file = os.path.join(RECORDINGS_DIR, "kinematics_summary.json")
        try:
            summary_dict = asdict(profile)
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_dict, f, indent=2)
            logger.info("💾 Generated kinematics summary: %s", summary_file)
        except Exception as e:
            logger.error("Failed to write summary: %s", e)

        # 5. Generate Markdown Report
        report_file = os.path.join(RECORDINGS_DIR, "TRAINING_REPORT.md")
        try:
            md_text = KinematicsAnalyzer.generate_markdown_report(profile)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(md_text)
            logger.info("📄 Generated Training Report: %s", report_file)
        except Exception as e:
            logger.error("Failed to write markdown report: %s", e)

        # 6. Apply to HandScroller profile & references directly
        KinematicsAnalyzer.apply_profile_to_disk(profile, PROFILE_PATH, REFERENCES_PATH)
        logger.info("🎯 Applied new kinematics calibration directly to HandScroller profile & references!")

        self.status_message = "🎉 TRAINING COMPLETE! Calibrated parameters applied. Press Q to exit."

        if self.on_session_completed:
            self.on_session_completed(profile)

    def get_hud_info(self, has_hand: bool = True) -> Dict[str, Any]:
        """Provides state for rendering the visual Training Cockpit HUD."""
        current_step_info = self.GUIDED_STEPS[self.step_idx] if (self.mode == self.MODE_GUIDED and self.step_idx < len(self.GUIDED_STEPS)) else {}
        return {
            'active': self.active,
            'mode': self.mode,
            'has_hand': has_hand,
            'session_id': self.session_id,
            'step_idx': self.step_idx,
            'total_steps': len(self.GUIDED_STEPS) if self.mode == self.MODE_GUIDED else 0,
            'step_title': current_step_info.get('title', ''),
            'step_desc': current_step_info.get('desc', ''),
            'status_message': self.status_message,
            'hold_progress': self.hold_progress,
            'in_transition': self.in_transition,
            'waiting_for_release': self.waiting_for_release,
            'is_recording_stroke': self.is_recording_stroke,
            'stroke_peak_disp': self.stroke_peak_disp,
            'stroke_direction': self.stroke_direction,
            'completed_count': len(self.completed_movements),
            'oscilloscope': list(self.oscilloscope_buffer),
            'drill_target_direction': self.drill_target_direction,
            'drill_count': self.drill_count,
            'drill_total': self.drill_target_total,
            'key_held': is_space_or_enter_held(),
        }
