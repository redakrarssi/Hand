"""
Touch Gesture Detector Module
Detects physical contact (touch) between:
- Big finger (Thumb tip, Landmark 4) and Index finger tip (Landmark 8) -> Scroll DOWN
- Big finger (Thumb tip, Landmark 4) and Middle finger tip (Landmark 12) -> Scroll UP

Features:
- Anatomical scale normalization (camera-distance invariant)
- Dual-threshold hysteresis (TOUCH_ENTER vs TOUCH_EXIT) to eliminate jitter
- Zero-latency instant firing on initial touch contact frame
- Hold-to-continuous-scroll (repeat after 0.32s at 0.14s interval)
- Real-time telemetry for HUD visualization (touch lines, glowing rings, proximity)
"""

import math
import time
import logging

logger = logging.getLogger(__name__)

# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_DIP = 11
MIDDLE_TIP = 12


def _get_point(pt):
    """Safely extract (x, y, z) coordinates from various point formats."""
    if hasattr(pt, 'x') and hasattr(pt, 'y'):
        return float(pt.x), float(pt.y), float(getattr(pt, 'z', 0.0))
    elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
        z = float(pt[2]) if len(pt) > 2 else 0.0
        return float(pt[0]), float(pt[1]), z
    return 0.0, 0.0, 0.0


def _dist_point_to_segment_2d(px, py, ax, ay, bx, by):
    """Calculates minimum 2D distance from point (px, py) to line segment [(ax, ay), (bx, by)]."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-7:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class TouchGestureDetector:
    """
    Detects physical touch between the thumb (big finger) and index or middle fingers.
    """

    ONSET_CONFIRM_FRAMES = 3      # Contact must hold ~100ms; a near-miss gap won't count
    RELEASE_CONFIRM_FRAMES = 2    # Ignore single-frame dropouts while pinching
    MIN_FIRE_INTERVAL = 0.30      # Responsive paging, never a skipped burst
    SETTLE_AFTER_FIRE = 0.55      # Ignore swipe motion right after a pinch
    AMBIGUITY_MARGIN = 0.025      # Clear distinction without penalizing natural adjacent finger curl

    def __init__(
        self,
        base_touch_threshold=0.24,
        base_release_threshold=0.34,
        hold_delay=1.2,
        repeat_interval=1.0,
        sensitivity=1.0,
    ):
        self.base_touch_threshold = base_touch_threshold
        self.base_release_threshold = base_release_threshold
        self.hold_delay = hold_delay
        self.repeat_interval = repeat_interval
        self.sensitivity = sensitivity

        # Current active touch state: None, 'index', or 'middle'
        self.active_touch = None
        self.touch_start_time = 0.0
        self.last_fire_time = -math.inf
        self._fired = False
        self._candidate = None
        self._candidate_frames = 0
        self._candidate_dropouts = 0
        self._release_frames = 0

        # Telemetry for HUD feedback and inspection
        self.dist_thumb_index = 1.0
        self.dist_thumb_middle = 1.0
        self.touching_index = False
        self.touching_middle = False
        self.thumb_pos = (0.0, 0.0)
        self.index_pos = (0.0, 0.0)
        self.middle_pos = (0.0, 0.0)
        self.hand_scale = 0.15

    def set_sensitivity(self, sensitivity):
        """Tune touch thresholds based on sensitivity setting."""
        if not math.isfinite(sensitivity):
            return
        self.sensitivity = max(0.3, min(3.0, sensitivity))

    @property
    def touch_threshold(self):
        """Dynamic touch threshold adjusted by sensitivity."""
        scaled = self.base_touch_threshold * (1.0 + 0.20 * (self.sensitivity - 1.0))
        return max(0.14, min(0.28, scaled))

    @property
    def release_threshold(self):
        """Dynamic release threshold (hysteresis band)."""
        scaled = self.base_release_threshold * (1.0 + 0.20 * (self.sensitivity - 1.0))
        return max(self.touch_threshold + 0.06, min(0.42, scaled))

    def reset(self):
        """Reset touch tracking state."""
        self.active_touch = None
        self.touch_start_time = 0.0
        self._fired = False
        self._candidate = None
        self._candidate_frames = 0
        self._candidate_dropouts = 0
        self._release_frames = 0
        self.touching_index = False
        self.touching_middle = False

    def update(self, landmarks, now=None):
        """
        Process the current hand landmarks to detect thumb-index and thumb-middle contact.
        
        Args:
            landmarks: List of MediaPipe landmarks (or objects with .x, .y, .z)
            now: Current timestamp (float seconds, defaults to monotonic time)
            
        Returns:
            GestureResult if a touch action should trigger, or None
        """
        if now is None:
            now = time.monotonic()

        if not landmarks or len(landmarks) < 13:
            self.reset()
            return None

        # Extract fingertip positions
        tx, ty, tz = _get_point(landmarks[THUMB_TIP])
        ix, iy, iz = _get_point(landmarks[INDEX_TIP])
        mx, my, mz = _get_point(landmarks[MIDDLE_TIP])
        wx, wy, wz = _get_point(landmarks[WRIST])
        kn_x, kn_y, kn_z = _get_point(landmarks[MIDDLE_MCP])

        # Distal joint positions for anatomical pad-to-pad & pad-to-tip contact
        tip_x, tip_y, tip_z = _get_point(landmarks[THUMB_IP])
        idip_x, idip_y, idip_z = _get_point(landmarks[INDEX_DIP])
        mdip_x, mdip_y, mdip_z = _get_point(landmarks[MIDDLE_DIP])

        # Save 2D positions for HUD display
        self.thumb_pos = (tx, ty)
        self.index_pos = (ix, iy)
        self.middle_pos = (mx, my)

        # Calculate hand scale (Wrist to Middle Knuckle MCP)
        hand_scale = math.hypot(kn_x - wx, kn_y - wy)
        if hand_scale < 0.05:
            # Fallback to Index MCP
            i_mcp_x, i_mcp_y, _ = _get_point(landmarks[INDEX_MCP])
            hand_scale = math.hypot(i_mcp_x - wx, i_mcp_y - wy)
            if hand_scale < 0.05:
                hand_scale = 0.15
        self.hand_scale = hand_scale

        # Normalized Euclidean distance with dampened depth noise.
        # Monocular depth Z from webcams has high variance; 2D screen distance is primary.
        raw_ti = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2 + ((tz - iz) * 0.20) ** 2)
        raw_tm = math.sqrt((tx - mx) ** 2 + (ty - my) ** 2 + ((tz - mz) * 0.20) ** 2)

        # Distal pad-to-pad & pad-to-tip minimum distance for real hands:
        # Check distal segment length (real hands have distal segments 0.08 to 0.45 of hand_scale)
        thumb_seg = math.hypot(tx - tip_x, ty - tip_y)
        index_seg = math.hypot(ix - idip_x, iy - idip_y)
        middle_seg = math.hypot(mx - mdip_x, my - mdip_y)

        if 0.08 * hand_scale <= thumb_seg <= 0.45 * hand_scale:
            if 0.08 * hand_scale <= index_seg <= 0.45 * hand_scale:
                d_t_to_ipad = _dist_point_to_segment_2d(tx, ty, idip_x, idip_y, ix, iy)
                d_i_to_tpad = _dist_point_to_segment_2d(ix, iy, tip_x, tip_y, tx, ty)
                raw_ti = min(raw_ti, d_t_to_ipad, d_i_to_tpad)

            if 0.08 * hand_scale <= middle_seg <= 0.45 * hand_scale:
                d_t_to_mpad = _dist_point_to_segment_2d(tx, ty, mdip_x, mdip_y, mx, my)
                d_m_to_tpad = _dist_point_to_segment_2d(mx, my, tip_x, tip_y, tx, ty)
                raw_tm = min(raw_tm, d_t_to_mpad, d_m_to_tpad)

        norm_ti = raw_ti / hand_scale
        self.dist_thumb_index = norm_ti

        norm_tm = raw_tm / hand_scale
        self.dist_thumb_middle = norm_tm

        touch_thresh = self.touch_threshold
        release_thresh = self.release_threshold

        trigger_gesture = None

        # Which fingertip is the thumb closing on?
        # Requires clear separation (AMBIGUITY_MARGIN) so curling adjacent fingers don't cause false scrolls
        if norm_ti <= touch_thresh and norm_ti + self.AMBIGUITY_MARGIN <= norm_tm:
            candidate = 'index'
        elif norm_tm <= touch_thresh and norm_tm + self.AMBIGUITY_MARGIN <= norm_ti:
            candidate = 'middle'
        else:
            candidate = None

        # ─── Touch State Machine: debounce, hysteresis, one scroll per pinch ───
        if self.active_touch is not None:
            norm_active = norm_ti if self.active_touch == 'index' else norm_tm
            if norm_active < release_thresh:
                self._release_frames = 0
                held = now - self.touch_start_time
                if not self._fired:
                    # Touch contact occurred during cooldown lockout: fire as soon as interval elapses
                    if now - self.last_fire_time >= self.MIN_FIRE_INTERVAL:
                        trigger_gesture = self._fire(self.active_touch, now)
                elif held >= self.hold_delay and now - self.last_fire_time >= self.repeat_interval:
                    # Slow, deliberate repeat only for a sustained hold (long feeds)
                    trigger_gesture = self._fire(self.active_touch, now)
            else:
                self._release_frames += 1
                if self._release_frames >= self.RELEASE_CONFIRM_FRAMES:
                    self.active_touch = None
                    self._release_frames = 0
                    self._candidate = None
                    self._candidate_frames = 0
                    self._candidate_dropouts = 0
            self.touching_index = self.active_touch == 'index'
            self.touching_middle = self.active_touch == 'middle'
        else:
            self.touching_index = self.touching_middle = False
            if candidate is not None:
                if candidate == self._candidate:
                    self._candidate_frames += 1
                else:
                    self._candidate = candidate
                    self._candidate_frames = 1
                self._candidate_dropouts = 0
            else:
                # 1 grace frame for monocular camera noise before wiping candidate progress
                if self._candidate_frames > 0 and self._candidate_dropouts < 1:
                    self._candidate_dropouts += 1
                else:
                    self._candidate = None
                    self._candidate_frames = 0
                    self._candidate_dropouts = 0

            if candidate is not None and self._candidate_frames >= self.ONSET_CONFIRM_FRAMES:
                self.active_touch = candidate
                self.touching_index = candidate == 'index'
                self.touching_middle = candidate == 'middle'
                self.touch_start_time = now
                self._release_frames = 0
                if now - self.last_fire_time >= self.MIN_FIRE_INTERVAL:
                    trigger_gesture = self._fire(candidate, now)
                else:
                    # Pinch began during lockout cooldown; will fire as soon as interval elapses
                    self._fired = False

        return trigger_gesture

    def _fire(self, finger, now):
        from app.gesture_engine import GestureResult
        self.last_fire_time = now
        self._fired = True
        if finger == 'index':
            return GestureResult('SWIPE_DOWN', confidence=1.0, velocity=1.0,
                                 direction=(0, 1), source='touch_thumb_index')
        return GestureResult('SWIPE_UP', confidence=1.0, velocity=1.0,
                             direction=(0, -1), source='touch_thumb_middle')

    def is_engaged(self, now):
        """True while the hand is pinching, closing in on a pinch, or just paged.

        The engine uses this to ignore the finger motion that leads into and out
        of a pinch, which otherwise registers as a second (swipe) scroll.
        """
        if self.active_touch is not None:
            return True
        if now - self.last_fire_time < self.SETTLE_AFTER_FIRE:
            return True
        # Only engaged if fingers are in immediate contact range (not when merely relaxed/hovering)
        near = self.touch_threshold * 1.05
        return min(self.dist_thumb_index, self.dist_thumb_middle) < near

    def get_telemetry(self):
        """Return current touch metrics dictionary for UI overlay and debugging."""
        return {
            'active_touch': self.active_touch,
            'touching_index': self.touching_index,
            'touching_middle': self.touching_middle,
            'dist_thumb_index': self.dist_thumb_index,
            'dist_thumb_middle': self.dist_thumb_middle,
            'thumb_pos': self.thumb_pos,
            'index_pos': self.index_pos,
            'middle_pos': self.middle_pos,
            'hand_scale': self.hand_scale,
            'touch_threshold': self.touch_threshold,
            'release_threshold': self.release_threshold,
        }
