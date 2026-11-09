"""
UI Overlay Module
Draws beautiful visual feedback on the camera preview window.
Shows hand landmarks, gesture indicators, status info, and controls.
"""

import cv2
import numpy as np
import time
import math
from app.config import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_SUCCESS,
    COLOR_WARNING, COLOR_TEXT, COLOR_BG, COLOR_LANDMARK, COLOR_CONNECTION,
    SHOW_LANDMARKS, SHOW_FPS, SHOW_GESTURE_FEEDBACK, OVERLAY_OPACITY,
)


class UIOverlay:
    """Draws visual overlay on camera frames for user feedback."""

    def __init__(self):
        self.gesture_flash = None       # Current gesture to display
        self.gesture_flash_time = 0     # When the gesture was triggered
        self.gesture_flash_duration = 0.8  # How long to show gesture
        self.fps = 0
        self.fps_update_time = 0
        self.frame_count = 0
        self._last_fps_time = time.time()
        self._fps_frames = 0
        self.status_message = "Starting..."
        self.is_paused = False
        self._smoothed_landmarks = None

    def update_fps(self):
        """Calculate running FPS."""
        now = time.time()
        self._fps_frames += 1
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._last_fps_time = now

    def trigger_gesture_flash(self, gesture_type, description="", success=True):
        """Trigger a visual gesture indicator."""
        self.gesture_flash = {
            'type': gesture_type,
            'description': description,
            'success': success,
        }
        self.gesture_flash_time = time.time()

    def _draw_rounded_rect(self, frame, pt1, pt2, color, thickness=-1, radius=10, alpha=0.7):
        """Draw a rounded rectangle with transparency."""
        overlay = frame.copy()
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw filled rounded rect on overlay
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_arrow(self, frame, center, direction, size=60, color=COLOR_PRIMARY, thickness=3):
        """Draw a directional arrow for gesture feedback."""
        cx, cy = center
        
        if direction == "SWIPE_DOWN":
            start = (cx, cy - size // 2)
            end = (cx, cy + size // 2)
        elif direction == "SWIPE_UP":
            start = (cx, cy + size // 2)
            end = (cx, cy - size // 2)
        elif direction == "SWIPE_LEFT":
            start = (cx + size // 2, cy)
            end = (cx - size // 2, cy)
        elif direction == "SWIPE_RIGHT":
            start = (cx - size // 2, cy)
            end = (cx + size // 2, cy)
        else:
            return

        cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.4)

    def _draw_gesture_feedback(self, frame):
        """Draw gesture detection feedback overlay."""
        if self.gesture_flash is None:
            return

        elapsed = time.time() - self.gesture_flash_time
        if elapsed > self.gesture_flash_duration:
            self.gesture_flash = None
            return

        h, w = frame.shape[:2]
        
        # Fade out effect
        alpha = 1.0 - (elapsed / self.gesture_flash_duration)
        alpha = max(0.0, min(1.0, alpha))

        gesture = self.gesture_flash
        gesture_type = gesture['type']
        description = gesture.get('description', gesture_type)

        # Color based on gesture type (SWIPE_UP is primary Next Video action)
        colors = {
            'SWIPE_UP': COLOR_SUCCESS,
            'SWIPE_DOWN': COLOR_PRIMARY,
            'SWIPE_LEFT': COLOR_WARNING,
            'SWIPE_RIGHT': COLOR_SECONDARY,
            'PALM_HOLD': COLOR_ACCENT,
            'NO_NO': (80, 200, 255),
        }
        color = colors.get(gesture_type, COLOR_TEXT)
        if not gesture.get('success', True):
            color = COLOR_ACCENT

        # Scale effect (grows then shrinks)
        scale = 1.0 + 0.3 * math.sin(elapsed * math.pi / self.gesture_flash_duration)

        # Draw large arrow in center
        arrow_size = int(80 * scale)
        center = (w // 2, h // 2)
        
        # Background circle
        overlay = frame.copy()
        cv2.circle(overlay, center, arrow_size + 20, (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)
        
        # Arrow
        if (gesture_type.startswith("SWIPE") or gesture_type.startswith("TOUCH")) and gesture.get('success', True):
            arrow_dir = "SWIPE_DOWN" if ("DOWN" in gesture_type) else ("SWIPE_UP" if ("UP" in gesture_type) else gesture_type)
            self._draw_arrow(frame, center, arrow_dir, arrow_size, color, 4)

        # Circle for palm hold
        if gesture_type == "PALM_HOLD":
            cv2.circle(frame, center, int(40 * scale), color, 3)
            cv2.putText(frame, "||", (center[0] - 15, center[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # Icon for NO_NO gesture
        if gesture_type == "NO_NO":
            cv2.circle(frame, center, int(45 * scale), (60, 200, 255), 3)
            cv2.putText(frame, "NO!", (center[0] - 32, center[1] + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 240, 255), 3)

        # Description text (dynamically sized to fit card)
        font_scale = 0.52 if len(description) > 35 else 0.75
        thickness = 2 if font_scale > 0.6 else 1
        text_size = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = max(10, (w - text_size[0]) // 2)
        text_y = center[1] + arrow_size + 38
        
        # Text background
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (text_x - 10, text_y - 22),
                     (text_x + text_size[0] + 10, text_y + 8),
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay2, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        
        cv2.putText(frame, description, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _draw_status_bar(self, frame, hand_info, brightness_info, input_stats, privacy_mode=False):
        """Draw top status bar with key information."""
        h, w = frame.shape[:2]
        bar_height = 40

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Left: App name + status
        if input_stats and not input_stats.get('target_ready', True):
            status_color = COLOR_WARNING
            status_text = "FOCUS BROWSER"
        elif privacy_mode:
            status_text = "HAND ACTIVE" if hand_info.get('tracking') else "PRIVACY MODE"
            status_color = COLOR_SUCCESS if hand_info.get('tracking') else COLOR_ACCENT
        else:
            status_text = "TRACKING" if hand_info.get('tracking') else "SEARCHING..."
            status_color = COLOR_SUCCESS if hand_info.get('tracking') else COLOR_ACCENT
        
        if self.is_paused:
            status_color = COLOR_WARNING
            status_text = "PAUSED"

        cv2.circle(frame, (15, bar_height // 2), 6, status_color, -1)
        cv2.putText(frame, f"HandScroller | {status_text}", (28, bar_height // 2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)

        # Center: Mode
        if input_stats:
            mode_text = f"Mode: {input_stats.get('effective_mode', 'scroll').upper()}"
            if hand_info.get('has_references'):
                mode_text += " | 🎯 6-REF SYNCED"
            mode_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, mode_text, ((w - mode_size[0]) // 2, bar_height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_SECONDARY, 1)

        # Right: FPS + Lighting Status + Video toggle hint
        b_name = brightness_info[0] if brightness_info else "Good"
        video_tag = "Live Cam" if not privacy_mode else "V: Preview"
        right_text = f"FPS: {self.fps:.0f} | {b_name} | {video_tag}"
        
        text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0]
        cv2.putText(frame, right_text, (w - text_size[0] - 10, bar_height // 2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT, 1)

        # Divider line
        cv2.line(frame, (0, bar_height), (w, bar_height), COLOR_PRIMARY, 1)

    def _draw_bottom_bar(self, frame, action_count=0):
        """Draw bottom control hints bar."""
        h, w = frame.shape[:2]
        bar_height = 32
        bar_y = h - bar_height

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Divider
        cv2.line(frame, (0, bar_y), (w, bar_y), (60, 60, 60), 1)

        # Controls text
        controls = "Q:Quit P:Pause R:Record | Touch: Thumb+Index v DOWN, Thumb+Mid ^ UP"
        cv2.putText(frame, controls, (10, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        # Action counter
        if action_count > 0:
            count_text = f"Actions: {action_count}"
            count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, count_text, (w - count_size[0] - 10, bar_y - 48),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_SUCCESS, 1)

    def _draw_hand_landmarks_custom(self, frame, landmarks_list, hand_info=None):
        """Draw minimal directional hand points focused strictly on the important scrolling fingers.
        
        Minimizes points to the essential 13 longitudinal spine points along the direction
        of scrolling (Index, Middle, Thumb, Wrist). Eliminates non-important fingers (Ring, Pinky)
        and renders dynamic scroll direction vectors at active fingertips.
        
        Args:
            frame: BGR frame to draw on
            landmarks_list: List of landmark lists from HandLandmarker
            hand_info: Optional hand tracking info containing telemetry and intent
        """
        if not landmarks_list and not (hand_info or {}).get('dense_landmarks'):
            self._smoothed_landmarks = None
            return

        h, w = frame.shape[:2]

        # Extract base 21 landmarks (from landmarks_list or fallback)
        landmarks = landmarks_list[0] if (landmarks_list and len(landmarks_list) > 0) else None
        if landmarks is None:
            self._smoothed_landmarks = None
            return

        # Smooth landmarks for jitter-free visual precision
        if self._smoothed_landmarks is None or len(self._smoothed_landmarks) != len(landmarks):
            self._smoothed_landmarks = [(p.x, p.y) for p in landmarks]
        else:
            alpha = 0.72  # Responsive and fluid
            self._smoothed_landmarks = [
                (alpha * p.x + (1 - alpha) * s[0], alpha * p.y + (1 - alpha) * s[1])
                for p, s in zip(landmarks, self._smoothed_landmarks)
            ]

        pts_2d = [(int(p[0] * w), int(p[1] * h)) for p in self._smoothed_landmarks]

        # ── 1. Color Palette Driven by Scroll Direction & Micro-Intent ──
        is_resting = (hand_info or {}).get('is_resting', True)
        micro_intent = (hand_info or {}).get('micro_intent')
        touching_idx = (hand_info or {}).get('touching_index', False)
        touching_mid = (hand_info or {}).get('touching_middle', False)

        active_scroll_dir = None
        if touching_idx or micro_intent == 'DOWN':
            active_scroll_dir = 'DOWN'
            spine_color = (60, 180, 255)       # Electric amber/coral
            tip_color = (50, 195, 255)
        elif touching_mid or micro_intent == 'UP':
            active_scroll_dir = 'UP'
            spine_color = (100, 240, 160)      # Neon emerald green
            tip_color = (120, 255, 80)
        elif is_resting:
            spine_color = (80, 210, 160)       # Calm mint/cyan
            tip_color = (70, 230, 150)
        else:
            spine_color = (180, 215, 80)       # Active cyan/lime
            tip_color = (255, 210, 60)

        # ── 2. Minimal Directional Connections Along Direction of Scrolling ──
        # Strictly the 3 important fingers (Index, Middle, Thumb) + Wrist anchor.
        # Ring (13-16) and Pinky (17-20) are excluded to eliminate clutter and noise.
        scroll_spine_connections = [
            # Index finger vertical scrolling spine (⬇ Primary DOWN / stroke)
            (5, 6), (6, 7), (7, 8),
            # Middle finger vertical scrolling spine (⬆ Primary UP / stroke)
            (9, 10), (10, 11), (11, 12),
            # Thumb pinch spine
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Palm structural base anchor
            (0, 5), (0, 9), (5, 9),
        ]

        # Draw longitudinal directional spine lines
        for start_idx, end_idx in scroll_spine_connections:
            if start_idx < len(pts_2d) and end_idx < len(pts_2d):
                p1 = pts_2d[start_idx]
                p2 = pts_2d[end_idx]
                cv2.line(frame, p1, p2, spine_color, 2, cv2.LINE_AA)

        # ── 3. Minimal Essential Landmark Nodes (13 Points Only) ──
        # Thumb: 1, 2, 3, 4 | Index: 5, 6, 7, 8 | Middle: 9, 10, 11, 12 | Wrist: 0
        important_landmarks = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        for i in important_landmarks:
            if i < len(pts_2d):
                x, y = pts_2d[i]
                if i in (4, 8, 12):  # Active fingertips
                    cv2.circle(frame, (x, y), 6, tip_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
                elif i in (0, 5, 9):  # Structural knuckles & wrist
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 3, spine_color, -1, cv2.LINE_AA)
                else:  # Intermediate phalanx joints
                    cv2.circle(frame, (x, y), 3, tip_color, -1, cv2.LINE_AA)

        # ── 4. Dynamic Scrolling Direction Vectors on Important Fingertips ──
        # Shows directional flow arrows along the Index (8) and Middle (12) tips
        pulse = 1.0 + 0.25 * math.sin(time.time() * 7)
        for tip_idx in (8, 12):
            if tip_idx < len(pts_2d):
                tx, ty = pts_2d[tip_idx]
                arrow_len = int(20 * pulse)

                if active_scroll_dir == 'UP':
                    # Direction of scrolling: UP ⬆️
                    cv2.arrowedLine(frame, (tx, ty + 6), (tx, ty - arrow_len),
                                    tip_color, 2, tipLength=0.45)
                elif active_scroll_dir == 'DOWN':
                    # Direction of scrolling: DOWN ⬇️
                    cv2.arrowedLine(frame, (tx, ty - 6), (tx, ty + arrow_len),
                                    tip_color, 2, tipLength=0.45)
                else:
                    # Neutral readiness ticks aligned with vertical scroll axis
                    cv2.line(frame, (tx, ty - 7), (tx, ty + 7), tip_color, 1, cv2.LINE_AA)
                    cv2.line(frame, (tx - 3, ty - 7), (tx + 3, ty - 7), tip_color, 1, cv2.LINE_AA)
                    cv2.line(frame, (tx - 3, ty + 7), (tx + 3, ty + 7), tip_color, 1, cv2.LINE_AA)

        # ── 5. Tactile Halo on Important Fingertips ──
        halo_pulse = 1.0 + 0.15 * math.sin(time.time() * 6)
        for tip_idx in (4, 8, 12):
            if tip_idx < len(pts_2d):
                tx, ty = pts_2d[tip_idx]
                cv2.circle(frame, (tx, ty), int(10 * halo_pulse), tip_color, 1, cv2.LINE_AA)

        self._draw_touch_visuals(frame, hand_info)

    def _draw_touch_visuals(self, frame, hand_info):
        """Draw interactive tactile feedback between thumb and index/middle fingertips."""
        if not hand_info or not hand_info.get('tracking'):
            return

        h, w = frame.shape[:2]
        thumb_pos = hand_info.get('thumb_pos')
        index_pos = hand_info.get('index_pos')
        middle_pos = hand_info.get('middle_pos')
        if not thumb_pos or not index_pos or not middle_pos:
            return

        tx, ty = int(thumb_pos[0] * w), int(thumb_pos[1] * h)
        ix, iy = int(index_pos[0] * w), int(index_pos[1] * h)
        mx, my = int(middle_pos[0] * w), int(middle_pos[1] * h)

        touching_idx = hand_info.get('touching_index', False)
        touching_mid = hand_info.get('touching_middle', False)
        dist_ti = hand_info.get('dist_thumb_index', 1.0)
        dist_tm = hand_info.get('dist_thumb_middle', 1.0)
        thresh = hand_info.get('touch_threshold', 0.22)

        pulse = 0.5 + 0.5 * math.sin(time.time() * 8)

        # ── Thumb + Index Touch (Scroll DOWN) ──
        if touching_idx:
            # Active contact! Glowing electric amber/coral line & rings
            cv2.line(frame, (tx, ty), (ix, iy), (50, 180, 255), 4, cv2.LINE_AA)
            cv2.line(frame, (tx, ty), (ix, iy), (255, 255, 255), 1, cv2.LINE_AA)
            r = int(10 + 6 * pulse)
            cv2.circle(frame, (tx, ty), r, (50, 180, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), r, (50, 180, 255), 2, cv2.LINE_AA)
            # Contact point badge
            mid_x, mid_y = (tx + ix) // 2, (ty + iy) // 2
            cv2.circle(frame, (mid_x, mid_y), 13, (20, 20, 20), -1)
            cv2.circle(frame, (mid_x, mid_y), 13, (50, 180, 255), 2)
            cv2.putText(frame, "v", (mid_x - 4, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 255), 2)
        elif dist_ti < 0.30:
            # Proximity guide (only when genuinely approaching contact, never during resting poses)
            proximity_alpha = max(0.0, min(1.0, (0.30 - dist_ti) / max(0.01, 0.30 - thresh)))
            col_b = int(255 * proximity_alpha)
            cv2.line(frame, (tx, ty), (ix, iy), (col_b, int(150 * proximity_alpha), 50), 1, cv2.LINE_AA)

        # ── Thumb + Middle Touch (Scroll UP) ──
        if touching_mid:
            # Active contact! Glowing neon emerald green line & rings
            cv2.line(frame, (tx, ty), (mx, my), (80, 255, 120), 4, cv2.LINE_AA)
            cv2.line(frame, (tx, ty), (mx, my), (255, 255, 255), 1, cv2.LINE_AA)
            r = int(10 + 6 * pulse)
            cv2.circle(frame, (tx, ty), r, (80, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(frame, (mx, my), r, (80, 255, 120), 2, cv2.LINE_AA)
            # Contact point badge
            mid_x, mid_y = (tx + mx) // 2, (ty + my) // 2
            cv2.circle(frame, (mid_x, mid_y), 13, (20, 20, 20), -1)
            cv2.circle(frame, (mid_x, mid_y), 13, (80, 255, 120), 2)
            cv2.putText(frame, "^", (mid_x - 4, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 255, 150), 2)
        elif dist_tm < 0.30:
            # Proximity guide (only when genuinely approaching contact, never during resting poses)
            proximity_alpha = max(0.0, min(1.0, (0.30 - dist_tm) / max(0.01, 0.30 - thresh)))
            col_g = int(240 * proximity_alpha)
            cv2.line(frame, (tx, ty), (mx, my), (80, col_g, 100), 1, cv2.LINE_AA)

    def _draw_finger_indicator(self, frame, hand_info):
        """Draw streamlined finger state, resting indicator, and directional scroll metrics."""
        if not hand_info.get('tracking'):
            return

        h, w = frame.shape[:2]
        x_start = w - 175
        y_start = 48
        
        # Background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 6, y_start - 4),
                     (w - 10, y_start + 72), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x_start - 6, y_start - 4), (w - 10, y_start + 72), (55, 55, 55), 1)

        # Header: Directional tracking focus on Important Fingers only
        cv2.putText(frame, "SCROLL ENGINE (3 FINGERS)", (x_start, y_start + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (100, 220, 255), 1)

        # Important fingers: Thumb (T), Index (I), Middle (M)
        finger_labels = [("T", "Touch"), ("I", "v DOWN"), ("M", "^ UP")]
        fingers = hand_info.get('fingers_up', [False] * 5)[:3]
        
        for i, ((name, role), is_up) in enumerate(zip(finger_labels, fingers)):
            x = x_start + i * 54
            color = COLOR_SUCCESS if is_up else (85, 85, 85)
            cv2.circle(frame, (x + 8, y_start + 25), 5, color, -1)
            cv2.putText(frame, f"{name}: {role}", (x + 16, y_start + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.26, COLOR_TEXT, 1)

        # Resting / Touch / Tracking State indicator with Stability %
        touching_idx = hand_info.get('touching_index', False)
        touching_mid = hand_info.get('touching_middle', False)
        is_resting = hand_info.get('is_resting', False)
        micro_intent = hand_info.get('micro_intent')
        stability = int(hand_info.get('resting_stability', 1.0) * 100)

        if touching_idx:
            status_text = "TOUCH: T+I -> SCROLL DOWN"
            status_color = (50, 180, 255)
        elif touching_mid:
            status_text = "TOUCH: T+M -> SCROLL UP"
            status_color = (120, 255, 80)
        elif micro_intent == 'UP':
            status_text = "SCROLL TOP (Index/Mid ^)"
            status_color = (120, 255, 80)
        elif micro_intent == 'DOWN':
            status_text = "SCROLL BOTTOM (Index/Mid v)"
            status_color = (50, 180, 255)
        elif is_resting:
            status_text = f"RESTING ({stability}% stable)"
            status_color = COLOR_SUCCESS
        else:
            status_text = "RELAX HAND"
            status_color = (200, 200, 200)

        cv2.putText(frame, status_text, (x_start, y_start + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, status_color, 1)
        
        # Noise floor indicator
        noise_floor = hand_info.get('noise_floor', 0.0018)
        cv2.putText(frame, f"Sensor Noise: {noise_floor * 1000:.1f}px floor", (x_start, y_start + 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 140, 140), 1)

    def _draw_no_hand_message(self, frame, privacy_mode=False):
        """Draw a message and full wide-space framing boundaries when searching for hand."""
        h, w = frame.shape[:2]
        
        # Draw full wide-space boundary guides (showing entire camera space)
        margin_x, margin_y = 12, 48
        border_col = (45, 55, 75)
        # Subtle framing rectangle outlining full wide viewing area
        cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - 38), border_col, 1)

        # Corner reticles for wide tracking field-of-view visual
        corner_len = 16
        reticle_col = (60, 180, 240)
        for (cx, cy, dx, dy) in [
            (margin_x, margin_y, 1, 1),
            (w - margin_x, margin_y, -1, 1),
            (margin_x, h - 38, 1, -1),
            (w - margin_x, h - 38, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), reticle_col, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), reticle_col, 2)

        # Pulsating opacity
        pulse = 0.5 + 0.3 * math.sin(time.time() * 3)
        
        text = "WIDE-VIEW TRACKING ACTIVE"
        hint = "Hand can be anywhere in view | Press 'V' to preview camera & lighting"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 - 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 15, text_y - 28),
                     (text_x + text_size[0] + 15, text_y + 40),
                     (15, 15, 15), -1)
        cv2.addWeighted(overlay, pulse * 0.75, frame, 1 - pulse * 0.75, 0, frame)
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_SUCCESS if privacy_mode else COLOR_PRIMARY, 2)

        # Subtext hint
        hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.putText(frame, hint, 
                   ((w - hint_size[0]) // 2, text_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 210, 240), 1)

    def _draw_intent_hint(self, frame, hand_info, input_stats):
        h, w = frame.shape[:2]
        status = hand_info.get('intent_status', 'Starting...')
        if input_stats and not input_stats.get('target_ready', True):
            msg = 'CLICK YOUR BROWSER TO SCROLL (HUD is focused)'
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            box_x = max(10, (w - text_size[0]) // 2 - 10)
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, h - 110), (box_x + text_size[0] + 20, h - 85), (20, 40, 140), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, msg, (box_x + 10, h - 93), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 220, 255), 1)
        color = (50, 180, 255) if hand_info.get('touching_index') else ((120, 255, 80) if hand_info.get('touching_middle') else (COLOR_SUCCESS if hand_info.get('is_resting') else COLOR_WARNING))
        cv2.putText(frame, status, (12, h - 61), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
        mapping = "Touch Thumb+Index = DOWN v | Thumb+Middle = UP ^"
        cv2.putText(frame, mapping, (12, h - 43), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 220, 255), 1)
        if hand_info.get('cycle_state') == 'MOVING':
            progress = min(1.0, max(0.0, hand_info.get('intent_progress', 0)))
            cv2.rectangle(frame, (12, h - 78), (142, h - 73), (55, 55, 55), -1)
            cv2.rectangle(frame, (12, h - 78), (12 + int(progress * 130), h - 73), COLOR_PRIMARY, -1)

    def _draw_paused_overlay(self, frame):
        """Draw paused state overlay."""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        text = "PAUSED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WARNING, 3)
        
        hint = "Press P to resume"
        hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, hint, ((w - hint_size[0]) // 2, text_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    def _draw_learning_overlay(self, frame, learning_info):
        """Draw interactive learning curve wizard overlay with 10s countdown and sub-step badges."""
        h, w = frame.shape[:2]
        feedback = learning_info.get('feedback', '')
        progress = learning_info.get('progress', 0.0)
        state = learning_info.get('state', '')
        phase_title = learning_info.get('phase_title', 'Guided Calibration')
        sub_step = learning_info.get('sub_step', 1)
        time_rem = learning_info.get('time_remaining', 10)

        # Instruction card only: this never limits the detection area.
        card_w, card_h = min(530, w - 20), 180
        x1 = (w - card_w) // 2
        y1 = (h - card_h) // 2 - 10
        x2, y2 = x1 + card_w, y1 + card_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 16, 14), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Header Badge (Left)
        cv2.rectangle(frame, (x1 + 16, y1 + 12), (x1 + 230, y1 + 34), (40, 30, 20), -1)
        cv2.putText(frame, "🎓 GUIDED LEARNING MODE", (x1 + 22, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 2)

        # Countdown Timer Pill (Right)
        if "COMPLETE" not in state:
            timer_text = f"⏱️ {time_rem}s REMAINING"
            pill_w = 145
            pill_x1 = x2 - 16 - pill_w
            pill_y1 = y1 + 12
            pill_x2 = x2 - 16
            pill_y2 = y1 + 34
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (30, 50, 90), -1)
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (80, 200, 255), 1)
            cv2.putText(frame, timer_text, (pill_x1 + 10, pill_y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 230, 255), 1)

        # Phase title
        cv2.putText(frame, phase_title, (x1 + 16, y1 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 215, 80), 2)

        # Sub-step badges (Positions 1, 2, 3)
        badge_w = 110
        badge_gap = 10
        badge_start_x = x1 + 16
        badge_y1 = y1 + 64
        badge_y2 = y1 + 84

        prefix = "Relax" if "REST" in state else "Move"
        for i in range(1, 4):
            bx1 = badge_start_x + (i - 1) * (badge_w + badge_gap)
            bx2 = bx1 + badge_w
            if i == sub_step:
                bg_color = (40, 130, 50)
                border_color = (80, 240, 100)
                txt_color = (255, 255, 255)
                lbl = f"▶ {prefix} {i}"
            elif i < sub_step:
                bg_color = (20, 60, 25)
                border_color = (40, 120, 50)
                txt_color = (160, 220, 160)
                lbl = f"✓ {prefix} {i}"
            else:
                bg_color = (30, 30, 30)
                border_color = (55, 55, 55)
                txt_color = (130, 130, 130)
                lbl = f"   {prefix} {i}"

            cv2.rectangle(frame, (bx1, badge_y1), (bx2, badge_y2), bg_color, -1)
            cv2.rectangle(frame, (bx1, badge_y1), (bx2, badge_y2), border_color, 1)
            cv2.putText(frame, lbl, (bx1 + 14, badge_y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_color, 1)

        # Feedback instruction text
        cv2.putText(frame, feedback, (x1 + 16, y1 + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # Progress bar
        bar_x = x1 + 16
        bar_y = y1 + 124
        bar_w = card_w - 32
        bar_h = 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (50, 220, 100), -1)
            pct_text = f"{int(progress * 100)}%"
            cv2.putText(frame, pct_text, (bar_x + max(4, fill_w - 35), bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 10, 10), 1)

        # Hint text
        hint_str = "Learning mode is 10s per movement for accurate calibration. Press L to restart."
        cv2.putText(frame, hint_str, (x1 + 16, y1 + 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)

    def _draw_recorder_overlay(self, frame, recorder_info):
        """Draw interactive whole-movement reference recorder overlay."""
        if not recorder_info or not recorder_info.get('active'):
            return

        h, w = frame.shape[:2]
        step_idx = recorder_info.get('step_index', 1)
        total_steps = recorder_info.get('total_steps', 7)
        title = recorder_info.get('title', 'Reference Calibration')
        subtitle = recorder_info.get('subtitle', '')
        status_msg = recorder_info.get('status_message', '')
        progress = recorder_info.get('hold_progress', 0.0)
        in_trans = recorder_info.get('in_transition', False)
        direction = recorder_info.get('direction', 'REST')
        is_moving = recorder_info.get('is_recording_stroke', False)
        stroke_disp = recorder_info.get('stroke_disp', 0.0)
        stroke_peak_disp = recorder_info.get('stroke_peak_disp', 0.0)
        last_mv = recorder_info.get('last_movement')

        key_held = recorder_info.get('key_held', False)
        waiting_rel = recorder_info.get('waiting_for_release', False)

        # Card size and position
        card_w, card_h = min(610, w - 24), 225
        x1 = (w - card_w) // 2
        y1 = 62
        x2, y2 = x1 + card_w, y1 + card_h

        # Semi-transparent dark background card (blend card area directly without cloning entire frame)
        card_sub = frame[y1:y2, x1:x2]
        dark_card = np.full_like(card_sub, 18)
        cv2.addWeighted(dark_card, 0.90, card_sub, 0.10, 0, card_sub)

        if in_trans:
            border_col = (50, 240, 100)
        elif waiting_rel:
            border_col = (80, 180, 255)
        elif is_moving or key_held:
            border_col = (80, 255, 255)
        elif direction == 'UP':
            border_col = (80, 220, 255)
        elif direction == 'DOWN':
            border_col = (255, 180, 50)
        else:
            border_col = (255, 215, 80)

        cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 2)

        # Header Badge: Flashing Red Record Dot + Title
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
        dot_color = (0, 0, int(155 + 100 * pulse)) if not (is_moving or key_held) else (0, int(200 + 55 * pulse), 255)
        cv2.circle(frame, (x1 + 24, y1 + 22), 7, dot_color, -1)
        if waiting_rel:
            badge_title = "RELEASE SPACE TO PROCEED TO NEXT MOVE"
            badge_col = (120, 220, 255)
        elif key_held or is_moving:
            badge_title = "RECORDING WHOLE MOVEMENT (SPACE HELD) - RELEASE WHEN DONE"
            badge_col = (100, 255, 255)
        else:
            badge_title = "WHOLE-MOVEMENT RECORDER (PUSH-TO-RECORD: HOLD SPACE)"
            badge_col = (255, 255, 255)
        cv2.putText(frame, badge_title, (x1 + 38, y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_col, 2)

        # Step count badge (Right)
        step_str = f"STEP {step_idx}/{total_steps}"
        cv2.rectangle(frame, (x2 - 110, y1 + 12), (x2 - 16, y1 + 34), (40, 50, 70), -1)
        cv2.putText(frame, step_str, (x2 - 100, y1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 255), 1)

        # Step Sequence Pills (7 steps: REST, UP1, UP2, UP3, DOWN1, DOWN2, DOWN3)
        pill_names = ["REST", "UP 1", "UP 2", "UP 3", "DOWN 1", "DOWN 2", "DOWN 3"]
        pill_gap = 6
        avail_w = card_w - 32
        pill_w = (avail_w - (len(pill_names) - 1) * pill_gap) // len(pill_names)
        pill_y1 = y1 + 44
        pill_y2 = pill_y1 + 24

        for i, name in enumerate(pill_names):
            px1 = x1 + 16 + i * (pill_w + pill_gap)
            px2 = px1 + pill_w
            curr_step_num = i + 1

            if curr_step_num == step_idx:
                p_bg = (40, 130, 60) if in_trans else ((20, 100, 140) if (is_moving or key_held) else (30, 80, 140))
                p_border = (80, 255, 120) if in_trans else ((100, 255, 255) if (is_moving or key_held) else (80, 200, 255))
                p_txt = (255, 255, 255)
                lbl = f">{name}"
            elif curr_step_num < step_idx:
                p_bg = (20, 60, 30)
                p_border = (40, 140, 60)
                p_txt = (140, 230, 160)
                lbl = f"V {name}"
            else:
                p_bg = (28, 28, 28)
                p_border = (50, 50, 50)
                p_txt = (120, 120, 120)
                lbl = name

            cv2.rectangle(frame, (px1, pill_y1), (px2, pill_y2), p_bg, -1)
            cv2.rectangle(frame, (px1, pill_y1), (px2, pill_y2), p_border, 1)
            cv2.putText(frame, lbl, (px1 + 4, pill_y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, p_txt, 1)

        # Main Step Title & Subtitle
        dir_color = (100, 240, 120) if direction == 'UP' else ((80, 180, 255) if direction == 'DOWN' else (255, 215, 80))
        cv2.putText(frame, title, (x1 + 16, y1 + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, dir_color, 2)
        cv2.putText(frame, subtitle, (x1 + 16, y1 + 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        # Dynamic Movement / Hold Meter Bar
        bar_x = x1 + 16
        bar_y = y1 + 130
        bar_w = card_w - 32
        bar_h = 24
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 35), -1)

        if waiting_rel:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 80, 140), -1)
            bar_txt = "PLEASE RELEASE SPACE TO GET READY FOR NEXT MOVE"
            txt_col = (255, 255, 255)

        elif direction == 'REST':
            # Resting baseline calibration fill
            fill_w = int(bar_w * max(0.0, min(1.0, progress)))
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (50, 220, 100), -1)
            if key_held:
                bar_txt = f"Calibrating baseline: {int(progress * 100)}% (Release when ready)"
            else:
                bar_txt = f"Hold neutral rest: {int(progress * 100)}% | Or HOLD SPACE for ~1s" if progress > 0 else "HOLD SPACE for ~1s while resting hand"
            txt_col = (255, 255, 255)

        elif in_trans:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 160, 70), -1)
            if last_mv:
                bar_txt = f"RECORDED! Peak: {last_mv['peak_displacement']*1000:.0f}px | Duration: {last_mv['duration']*1000:.0f}ms"
            else:
                bar_txt = "WHOLE MOVEMENT RECORDED & LOCKED!"
            txt_col = (255, 255, 255)

        elif is_moving or key_held:
            # Live stroke trajectory fill
            fill_ratio = min(1.0, max(0.0, stroke_disp / 0.050))
            fill_w = int(bar_w * fill_ratio)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (40, 200, 255), -1)
            bar_txt = f"[REC] {direction}: {stroke_disp*1000:.0f}px (Peak: {stroke_peak_disp*1000:.0f}px) - RELEASE SPACE TO FINISH"
            txt_col = (20, 20, 20) if fill_w > 200 else (255, 255, 255)

        else:
            # Waiting for stroke onset
            bar_txt = f"HOLD SPACE -> Move hand {direction} -> RELEASE SPACE when done"
            txt_col = (170, 210, 255)

        cv2.putText(frame, bar_txt, (bar_x + 12, bar_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1)

        # Status / Feedback message
        status_color = (100, 255, 120) if in_trans else ((100, 220, 255) if (is_moving or key_held) else (200, 200, 200))
        cv2.putText(frame, status_msg, (x1 + 16, y1 + 176),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1)

        # Instructions / Hints
        hint_text = "HOLD SPACE while doing hand movement -> RELEASE SPACE when done | ESC: cancel"
        cv2.putText(frame, hint_text, (x1 + 16, y1 + 202),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (160, 160, 160), 1)

    def draw(self, frame, landmarks_list, hand_info, brightness_info, input_stats, privacy_mode=False, learning_info=None, recorder_info=None):
        """
        Draw the complete UI overlay on the frame.
        """
        self.update_fps()

        # Draw hand landmarks
        if SHOW_LANDMARKS:
            self._draw_hand_landmarks_custom(frame, landmarks_list, hand_info=hand_info)

        # Draw status bar
        self._draw_status_bar(frame, hand_info, brightness_info, input_stats, privacy_mode=privacy_mode)

        total_pts = hand_info.get('total_points', 134)
        cv2.putText(frame, f"DENSE SENSING: {total_pts} PTS | Dominant Focus (Index: 32, Mid: 28, Thm: 26)", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 220, 255), 1)

        # Draw finger indicator
        self._draw_finger_indicator(frame, hand_info)

        # Draw gesture feedback
        if SHOW_GESTURE_FEEDBACK:
            self._draw_gesture_feedback(frame)

        # Draw bottom bar
        action_count = input_stats.get('total_actions', 0) if input_stats else 0
        self._draw_bottom_bar(frame, action_count)
        self._draw_intent_hint(frame, hand_info, input_stats)

        # Draw 6-position reference recorder overlay if active
        if recorder_info and recorder_info.get('active'):
            self._draw_recorder_overlay(frame, recorder_info)
        elif learning_info and learning_info.get('active'):
            self._draw_learning_overlay(frame, learning_info)

        # No hand message (suppress while recorder is active to prevent obscuring the card)
        if not hand_info.get('tracking') and not self.is_paused and not (recorder_info and recorder_info.get('active')):
            self._draw_no_hand_message(frame, privacy_mode=privacy_mode)

        # Paused overlay
        if self.is_paused:
            self._draw_paused_overlay(frame)

        return frame
