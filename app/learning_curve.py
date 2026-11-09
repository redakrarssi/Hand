"""
Learning Curve Module
Guided calibration wizard that learns the user's personal hand range,
natural stroke style, and relaxed posture, then saves a personalized profile
and automatically launches the interactive scroll test page.
"""

import json
import logging
import math
import os
import time
import webbrowser
from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional, Dict, Any

from app.intent_tracker import bounded_threshold
from app.config import INTENT_FINGER_DISTANCE, INTENT_HAND_DISTANCE

logger = logging.getLogger("HandScroller.Learning")

PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_profile.json")
TEST_PAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scroll_test.html")


@dataclass
class UserProfile:
    calibrated: bool = False
    finger_distance: float = 0.038
    hand_distance: float = 0.055
    sensitivity: float = 1.0
    phone_style: bool = True
    calibrated_at: float = 0.0
    down_stroke_recorded: float = 0.0
    up_stroke_recorded: float = 0.0
    corrections_count: int = 0
    relax_noise_floor: float = 0.008
    has_recorded_references: bool = False
    gesture_references: Optional[dict] = None


def load_profile():
    """Load existing user profile or return defaults."""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                valid_fields = {f.name for f in UserProfile.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in valid_fields}
                profile = UserProfile(**filtered)
                profile.finger_distance = bounded_threshold(profile.finger_distance, INTENT_FINGER_DISTANCE, 0.018, 0.065)
                profile.hand_distance = bounded_threshold(profile.hand_distance, INTENT_HAND_DISTANCE, 0.028, 0.085)
                return profile
        except Exception as e:
            logger.warning("Could not read profile: %s", e)
    return UserProfile()


def save_profile(profile: UserProfile):
    """Save user profile to disk."""
    try:
        with open(PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, indent=2)
        logger.info("Saved personalized user profile to %s", PROFILE_PATH)
        return True
    except Exception as e:
        logger.error("Failed to save profile: %s", e)
        return False


def open_test_page():
    """Open the requested test page without a delayed focus-stealing worker."""
    try:
        from pathlib import Path
        uri = Path(TEST_PAGE_PATH).resolve().as_uri()
        if not webbrowser.open(uri):
            logger.warning('The browser could not open the scroll test page')
            return False
        logger.info("🚀 Opened interactive scroll test page: %s", uri)
        return True
    except Exception as e:
        logger.warning("Could not open browser automatically: %s", e)
        return False


class LearningCurve:
    """
    Step-by-step interactive learning curve wizard.
    Phase 1: Relax Mode (10 sec total): Relax 1 (Center), Relax 2 (Comfort), Relax 3 (Noise floor)
    Phase 2: Scroll Down Mode (10 sec total): Move 1, Move 2, Move 3 natural down flicks
    Phase 3: Scroll Up Mode (10 sec total): Move 1, Move 2, Move 3 natural up flicks
    Phase 4: Save personalized profile, apply thresholds live, & open interactive test page
    """

    STATE_INACTIVE = "INACTIVE"
    STATE_REST = "LEARNING_REST"
    STATE_DOWN = "LEARNING_DOWN"
    STATE_UP = "LEARNING_UP"
    STATE_COMPLETE = "COMPLETE"

    def __init__(self, on_complete_callback=None, phase_duration=10.0):
        self.state = self.STATE_INACTIVE
        self.on_complete_callback = on_complete_callback
        self.phase_duration = float(phase_duration)
        self.step_start_time = None
        self.rest_samples = []
        self.down_displacements = []
        self.up_displacements = []
        self.down_flicks = []
        self.up_flicks = []
        self.in_stroke = False
        self.stroke_peak = 0.0
        self.baseline_y = None
        self.baseline_finger_y = None
        self.baseline_scale = None
        self.relax_noise_floor = 0.008
        self.step_feedback = ""
        self.phase_title = ""
        self.sub_step = 1
        self.sub_step_title = ""
        self.time_remaining = 10
        self.progress = 0.0
        self.profile = load_profile()

    def start(self):
        """Begin the learning curve process."""
        self.state = self.STATE_REST
        self.step_start_time = None
        self.rest_samples = []
        self.down_displacements = []
        self.up_displacements = []
        self.down_flicks = []
        self.up_flicks = []
        self.in_stroke = False
        self.stroke_peak = 0.0
        self.baseline_y = None
        self.baseline_finger_y = None
        self.baseline_scale = None
        self.relax_noise_floor = 0.008
        self.sub_step = 1
        self.phase_title = "Step 1 of 3: Relax Mode (10s)"
        self.sub_step_title = "Relax Mode 1/3: Comfortable Neutral"
        self.step_feedback = "Relax Mode 1: Rest hand anywhere in camera view (fingers relaxed)"
        self.time_remaining = int(self.phase_duration)
        self.progress = 0.0
        logger.info("🎓 Started HandScroller Guided Learning Mode (10s per phase)")

    def is_active(self):
        return self.state != self.STATE_INACTIVE

    def cancel(self):
        self.state = self.STATE_INACTIVE

    def update(self, landmarks, now):
        """
        Feed landmarks from the current frame into the learning wizard.
        Returns a dict of status to display on the HUD.
        """
        if self.state == self.STATE_INACTIVE:
            return None

        if self.step_start_time is None:
            self.step_start_time = now

        elapsed = max(0.0, now - self.step_start_time)
        sub_duration = self.phase_duration / 3.0
        self.time_remaining = max(0, int(math.ceil(self.phase_duration - elapsed)))

        if not landmarks:
            self.step_feedback = "Keep your hand visible in camera view"
            return self.get_hud_info()

        # Handle both 2D landmark list [[p0, ...]] from MediaPipe and 1D list [p0, ...]
        if isinstance(landmarks, (list, tuple)) and len(landmarks) > 0 and isinstance(landmarks[0], (list, tuple)):
            landmarks = landmarks[0]

        if len(landmarks) < 21:
            self.step_feedback = "Keep your hand visible in camera view"
            return self.get_hud_info()

        # Extract wrist (0), middle knuckle (9), and index tip (8)
        wrist = landmarks[0]
        knuckle = landmarks[9]
        index_tip = landmarks[8]
        scale = max(0.015, math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y))
        hand_y = (wrist.y + knuckle.y) / 2.0
        finger_y = index_tip.y

        # ─── PHASE 1: RELAX MODE (10.0 Seconds Total) ─────────────────
        if self.state == self.STATE_REST:
            self.rest_samples.append((hand_y, finger_y, scale))
            self.phase_title = f"Step 1 of 3: Relax Mode ({int(self.phase_duration)}s)"
            self.progress = min(1.0, elapsed / self.phase_duration)

            if elapsed < sub_duration:
                self.sub_step = 1
                self.sub_step_title = "Relax Mode 1/3: Comfortable Posture"
                self.step_feedback = "Relax Mode 1: Rest hand comfortably anywhere in view"
            elif elapsed < sub_duration * 2.0:
                self.sub_step = 2
                self.sub_step_title = "Relax Mode 2/3: Comfortable Rest"
                self.step_feedback = "Relax Mode 2: Rest hand at your natural desk angle"
            else:
                self.sub_step = 3
                self.sub_step_title = "Relax Mode 3/3: Measuring Noise Floor"
                self.step_feedback = "Relax Mode 3: Hold steady - sensors calibrating noise floor"

            if elapsed >= self.phase_duration and len(self.rest_samples) >= 15:
                # Calculate resting baseline from all 3 relaxed positions
                self.baseline_y = sum(s[0] for s in self.rest_samples) / len(self.rest_samples)
                self.baseline_finger_y = sum(s[1] for s in self.rest_samples) / len(self.rest_samples)
                self.baseline_scale = sum(s[2] for s in self.rest_samples) / len(self.rest_samples)

                # Measure noise floor across rest samples
                # Local frame differences measure jitter; changes between the
                # three comfortable postures must not become a huge noise floor.
                deviations = [abs(b[0] - a[0]) / max(0.015, (a[2] + b[2]) / 2)
                              for a, b in zip(self.rest_samples, self.rest_samples[1:])]
                self.relax_noise_floor = max(0.005, min(0.020, median(deviations) if deviations else 0.008))

                logger.info("✅ Baseline learned: scale=%.3f, hand_y=%.3f, noise_floor=%.4f",
                            self.baseline_scale, self.baseline_y, self.relax_noise_floor)

                # Advance to Phase 2: Scroll Down Mode
                self.state = self.STATE_DOWN
                self.step_start_time = now
                self.progress = 0.0
                self.sub_step = 1
                self.down_displacements = []
                self.down_flicks = []
                self.in_stroke = False
                self.stroke_peak = 0.0
                self.phase_title = f"Step 2 of 3: Teach Scroll Down ({int(self.phase_duration)}s)"
                self.sub_step_title = "Scroll Down Move 1/3"
                self.step_feedback = "Move 1/3: Flick hand/finger to SCROLL DOWN"

        # ─── PHASE 2: SCROLL DOWN MODE (10.0 Seconds Total) ───────────
        elif self.state == self.STATE_DOWN:
            self.phase_title = f"Step 2 of 3: Teach Scroll Down ({int(self.phase_duration)}s)"
            self.progress = min(1.0, elapsed / self.phase_duration)

            if elapsed < sub_duration:
                self.sub_step = 1
                self.sub_step_title = "Scroll Down Move 1/3"
            elif elapsed < sub_duration * 2.0:
                self.sub_step = 2
                self.sub_step_title = "Scroll Down Move 2/3"
            else:
                self.sub_step = 3
                self.sub_step_title = "Scroll Down Move 3/3"

            dy_up = max(0.0, (self.baseline_y - hand_y) / self.baseline_scale, (self.baseline_finger_y - finger_y) / self.baseline_scale)
            dy_down = max(0.0, (hand_y - self.baseline_y) / self.baseline_scale, (finger_y - self.baseline_finger_y) / self.baseline_scale)
            current_disp = max(dy_up, dy_down)

            if current_disp >= 0.024:
                self.in_stroke = True
                self.stroke_peak = max(self.stroke_peak, current_disp)
                self.down_displacements.append(current_disp)
                if dy_up > dy_down:
                    self.profile.phone_style = True   # Hand up = scroll down
                    dir_hint = "UP (Phone Style)"
                else:
                    self.profile.phone_style = False  # Hand down = scroll down
                    dir_hint = "DOWN (Natural Style)"
                self.step_feedback = f"Move {self.sub_step}/3 {dir_hint} captured! Return hand to rest..."
            else:
                if not self.in_stroke:
                    self.step_feedback = f"Move {self.sub_step}/3: Flick hand/finger to SCROLL DOWN"

            # When user returns hand to rest after stroke peak, record sample
            if self.in_stroke and current_disp < 0.020 and self.stroke_peak >= 0.028:
                self.down_flicks.append(self.stroke_peak)
                self.down_displacements.append(self.stroke_peak)
                self.in_stroke = False
                self.stroke_peak = 0.0
                logger.info("Recorded scroll DOWN flick sample %d: %.3f", len(self.down_flicks), self.down_flicks[-1])

            if elapsed >= self.phase_duration:
                recorded_down = max(self.down_flicks or self.down_displacements or [0.045])
                self.profile.down_stroke_recorded = recorded_down
                logger.info("✅ Scroll DOWN stroke learned: %.3f palm scale (phone_style=%s, samples=%d)",
                            recorded_down, self.profile.phone_style, len(self.down_flicks))

                # Advance to Phase 3: Scroll Up Mode
                self.state = self.STATE_UP
                self.step_start_time = now
                self.progress = 0.0
                self.sub_step = 1
                self.up_displacements = []
                self.up_flicks = []
                self.in_stroke = False
                self.stroke_peak = 0.0
                self.phase_title = f"Step 3 of 3: Teach Scroll Up ({int(self.phase_duration)}s)"
                self.sub_step_title = "Scroll Up Move 1/3"
                self.step_feedback = "Move 1/3: Flick in OPPOSITE direction for SCROLL UP"

        # ─── PHASE 3: SCROLL UP MODE (10.0 Seconds Total) ─────────────
        elif self.state == self.STATE_UP:
            self.phase_title = f"Step 3 of 3: Teach Scroll Up ({int(self.phase_duration)}s)"
            self.progress = min(1.0, elapsed / self.phase_duration)

            if elapsed < sub_duration:
                self.sub_step = 1
                self.sub_step_title = "Scroll Up Move 1/3"
            elif elapsed < sub_duration * 2.0:
                self.sub_step = 2
                self.sub_step_title = "Scroll Up Move 2/3"
            else:
                self.sub_step = 3
                self.sub_step_title = "Scroll Up Move 3/3"

            if self.profile.phone_style:
                dy = max(0.0, (hand_y - self.baseline_y) / self.baseline_scale, (finger_y - self.baseline_finger_y) / self.baseline_scale)
                dir_hint = "DOWN"
            else:
                dy = max(0.0, (self.baseline_y - hand_y) / self.baseline_scale, (self.baseline_finger_y - finger_y) / self.baseline_scale)
                dir_hint = "UP"

            if dy >= 0.024:
                self.in_stroke = True
                self.stroke_peak = max(self.stroke_peak, dy)
                self.up_displacements.append(dy)
                self.step_feedback = f"Move {self.sub_step}/3 {dir_hint} captured! Return hand to rest..."
            else:
                if not self.in_stroke:
                    self.step_feedback = f"Move {self.sub_step}/3: Flick {dir_hint} to SCROLL UP"

            # When user returns hand to rest after stroke peak, record sample
            if self.in_stroke and dy < 0.020 and self.stroke_peak >= 0.028:
                self.up_flicks.append(self.stroke_peak)
                self.up_displacements.append(self.stroke_peak)
                self.in_stroke = False
                self.stroke_peak = 0.0
                logger.info("Recorded scroll UP flick sample %d: %.3f", len(self.up_flicks), self.up_flicks[-1])

            if elapsed >= self.phase_duration:
                recorded_up = max(self.up_flicks or self.up_displacements or [0.045])
                self.profile.up_stroke_recorded = recorded_up
                logger.info("✅ Scroll UP stroke learned: %.3f palm scale (samples=%d)",
                            recorded_up, len(self.up_flicks))

                self._finalize_calibration(now)

        # ─── PHASE 4: FINALIZE & OPEN ─────────────────────────────────
        elif self.state == self.STATE_COMPLETE:
            if elapsed > 1.8:
                self.state = self.STATE_INACTIVE

        return self.get_hud_info()

    def _finalize_calibration(self, now=None):
        """Compute optimal parameters, save profile, and launch test page."""
        self.state = self.STATE_COMPLETE
        self.step_start_time = now
        self.progress = 1.0
        self.phase_title = "Calibration Complete!"
        self.sub_step_title = "Profile Saved"
        self.time_remaining = 0
        self.step_feedback = "🎉 Calibration Complete! 100% Synced to your hand."

        # Compute optimal personalized thresholds from recorded strokes
        down_val = sum(self.down_flicks) / len(self.down_flicks) if self.down_flicks else self.profile.down_stroke_recorded
        up_val = sum(self.up_flicks) / len(self.up_flicks) if self.up_flicks else self.profile.up_stroke_recorded
        min_stroke = min(down_val or 0.045, up_val or 0.045)
        noise_floor = getattr(self, 'relax_noise_floor', 0.008)

        # Trigger at 44% of natural flick for sensitive, snappy response,
        # while staying at least 1.8x above the relax noise floor so resting never triggers
        self.profile.finger_distance = bounded_threshold(max(noise_floor * 1.8, min_stroke * 0.44), INTENT_FINGER_DISTANCE, 0.018, 0.065)
        self.profile.hand_distance = bounded_threshold(max(noise_floor * 2.5, min_stroke * 0.65), INTENT_HAND_DISTANCE, 0.028, 0.085)
        self.profile.relax_noise_floor = min(0.020, noise_floor)
        self.profile.calibrated = True
        self.profile.calibrated_at = time.time()

        save_profile(self.profile)
        logger.info("🎉 Calibration complete! Finger threshold: %.3f, Hand: %.3f, PhoneStyle: %s",
                    self.profile.finger_distance, self.profile.hand_distance, self.profile.phone_style)

        if self.on_complete_callback:
            self.on_complete_callback(self.profile)

        # Open test page in default browser
        open_test_page()

    def get_hud_info(self):
        """Return information for UIOverlay rendering."""
        return {
            'active': self.state != self.STATE_INACTIVE,
            'state': self.state,
            'feedback': self.step_feedback,
            'progress': self.progress,
            'phase_title': getattr(self, 'phase_title', ''),
            'sub_step': getattr(self, 'sub_step', 1),
            'sub_step_title': getattr(self, 'sub_step_title', ''),
            'time_remaining': getattr(self, 'time_remaining', 10),
            'phone_style': self.profile.phone_style,
        }
