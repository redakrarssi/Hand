"""
╔══════════════════════════════════════════════════════════════════════╗
║               🖐️  HandScroller Training Station                      ║
║         Capture Hand Kinematics & Train the Perfect Code             ║
╚══════════════════════════════════════════════════════════════════════╝

An interactive workstation that records your real hand movements, analyzes
your personal motor kinematics, and generates the exact optimal code & thresholds.

Modes:
  [1] Guided Calibration : Rest baseline + 3 UP flicks + 3 DOWN flicks
  [2] Freeform Session   : Natural live scrolling (record while watching videos)
  [3] Repetition Drill   : 5x UP flicks, 5x DOWN flicks for variance & precision

Controls:
  SPACE     → Push-To-Record: Hold down while flicking, release when done
  1, 2, 3   → Select Mode (Guided, Freeform, Drill)
  V         → Toggle Camera Video (Privacy Canvas vs Real Video)
  M         → Toggle Mirror Mode
  A         → Instant Kinematic Analysis & Apply to HandScroller
  Q / ESC   → Save session & Quit
"""

import os
import sys
import time
import argparse
import logging
import math
from collections import deque
import cv2
import numpy as np

# Ensure HandScroller root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    os.chdir(SCRIPT_DIR)
except Exception:
    pass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from app.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CAMERA_BACKEND, CAMERA_EXPOSURE, CAMERA_AUTO_EXPOSURE,
    MIRROR_MODE, SHOW_CAMERA_FEED,
)
from app.camera_capture import LatestFrameCamera
from app.gesture_engine import GestureEngine, HAND_CONNECTIONS
from app.training_station_engine import TrainingStationEngine
from app.kinematics_analyzer import KinematicsAnalyzer, UserKinematicsProfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TrainingStation")

WINDOW_NAME = "HandScroller Training Station"
WIN_W = 860
WIN_H = 640


def parse_args():
    parser = argparse.ArgumentParser(description="HandScroller Training Station")
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX, help='Camera index')
    parser.add_argument('--mode', choices=['guided', 'freeform', 'drill'], default='guided', help='Initial training mode')
    parser.add_argument('--show-camera', action='store_true', help='Show real webcam video by default')
    return parser.parse_args()


def setup_camera(camera_index: int) -> cv2.VideoCapture:
    """Initialize camera with DirectShow/MSMF support on Windows."""
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened() and backend == cv2.CAP_DSHOW:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap


class MotionTrailVisualizer:
    """Renders a fading neon particle trail behind the index fingertip."""
    def __init__(self, max_points: int = 30):
        self.points = deque(maxlen=max_points)

    def add_point(self, pt: Tuple[int, int], velocity: float):
        self.points.append({'pt': pt, 'vel': abs(velocity), 't': time.time()})

    def draw(self, canvas: np.ndarray):
        now = time.time()
        pts_list = list(self.points)
        for i in range(1, len(pts_list)):
            p1 = pts_list[i - 1]['pt']
            p2 = pts_list[i]['pt']
            vel = pts_list[i]['vel']
            age = now - pts_list[i]['t']
            alpha = max(0.0, 1.0 - (age / 0.8))
            if alpha <= 0.05:
                continue

            # Speed-based color gradient:
            # Low: Emerald Green (50, 220, 100), Mid: Cyan/Orange, High: Bright Magenta (255, 60, 200)
            if vel > 0.40:
                color = (240, 50, 220)  # Magenta/Pink for rapid flick
                thickness = 4
            elif vel > 0.15:
                color = (80, 220, 255)  # Bright Cyan for movement
                thickness = 3
            else:
                color = (80, 220, 80)   # Green for rest
                thickness = 2

            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


class TrainingStationUI:
    """Renders high-visibility telemetry, oscilloscope waveforms, and instructions."""

    def __init__(self):
        self.motion_trail = MotionTrailVisualizer(max_points=25)
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_counter = 0

    def update_fps(self):
        self.fps_counter += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = now

    def render(
        self,
        canvas: np.ndarray,
        landmarks_list: Optional[Any],
        hud_info: Dict[str, Any],
        privacy_mode: bool = True
    ):
        self.update_fps()
        h, w = canvas.shape[:2]

        # Draw privacy background or dim video
        if privacy_mode:
            canvas[:] = 20  # Clean dark graphite canvas
        else:
            # Subtle dimming to ensure text readability
            cv2.addWeighted(canvas, 0.70, np.full_like(canvas, 15), 0.30, 0, canvas)

        # Draw Hand Landmarks Skeleton
        has_hand = hud_info.get('has_hand', False)
        if has_hand and landmarks_list:
            lms = landmarks_list[0] if (isinstance(landmarks_list, (list, tuple)) and len(landmarks_list) > 0 and isinstance(landmarks_list[0], (list, tuple))) else landmarks_list
            if len(lms) >= 21:
                self._draw_hand_skeleton(canvas, lms, hud_info)

        # Draw Motion Trail
        self.motion_trail.draw(canvas)

        # Draw Oscilloscope Waveform at the bottom
        self._draw_oscilloscope(canvas, hud_info.get('oscilloscope', []))

        # Draw Top Navigation & Status Bar
        self._draw_header(canvas, hud_info)

        # Draw Main Instruction Card (Center)
        self._draw_instruction_card(canvas, hud_info)

        # Draw Bottom Controls Bar
        self._draw_footer(canvas)

    def _draw_hand_skeleton(self, canvas: np.ndarray, landmarks, hud_info: Dict[str, Any]):
        h, w = canvas.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks[:21]]

        # Draw connections
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(canvas, pts[p1], pts[p2], (70, 90, 110), 2, cv2.LINE_AA)

        # Dominant finger highlights (Index=8, Middle=12, Thumb=4)
        for i, (x, y) in enumerate(pts):
            if i in (4, 8, 12):
                col = (80, 240, 255) if i == 8 else ((100, 220, 140) if i == 12 else (255, 180, 80))
                cv2.circle(canvas, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), 3, (160, 170, 180), -1, cv2.LINE_AA)

        # Add index tip to motion trail
        idx_pt = pts[8]
        osc = hud_info.get('oscilloscope', [])
        latest_vy = osc[-1]['vy'] if osc else 0.0
        self.motion_trail.add_point(idx_pt, latest_vy)

    def _draw_header(self, canvas: np.ndarray, hud_info: Dict[str, Any]):
        w = canvas.shape[1]
        cv2.rectangle(canvas, (0, 0), (w, 54), (28, 30, 36), -1)
        cv2.line(canvas, (0, 54), (w, 54), (55, 65, 80), 1)

        # App Title
        cv2.putText(canvas, "🖐️ HANDSCROLLER TRAINING STATION", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Mode Badge
        mode = hud_info.get('mode', 'GUIDED')
        badge_text = f"MODE: [{mode}]"
        cv2.rectangle(canvas, (w - 340, 12), (w - 180, 42), (45, 60, 90), -1)
        cv2.rectangle(canvas, (w - 340, 12), (w - 180, 42), (80, 180, 255), 1)
        cv2.putText(canvas, badge_text, (w - 330, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1, cv2.LINE_AA)

        # Hand Status Indicator
        has_hand = hud_info.get('has_hand', False)
        status_col = (80, 240, 120) if has_hand else (60, 60, 240)
        status_lbl = "HAND TRACKED" if has_hand else "NO HAND"
        cv2.circle(canvas, (w - 140, 27), 6, status_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, status_lbl, (w - 125, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1, cv2.LINE_AA)

    def _draw_instruction_card(self, canvas: np.ndarray, hud_info: Dict[str, Any]):
        h, w = canvas.shape[:2]
        mode = hud_info.get('mode', 'GUIDED')
        active = hud_info.get('active', False)
        key_held = hud_info.get('key_held', False)
        is_rec = hud_info.get('is_recording_stroke', False)
        in_trans = hud_info.get('in_transition', False)

        card_w = min(740, w - 40)
        card_h = 175
        x1 = (w - card_w) // 2
        y1 = 68
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Semi-transparent backing
        overlay = canvas[y1:y2, x1:x2].copy()
        cv2.rectangle(overlay, (0, 0), (card_w, card_h), (20, 24, 30), -1)
        border_col = (80, 255, 120) if in_trans else ((80, 240, 255) if (key_held or is_rec) else (60, 80, 110))
        cv2.addWeighted(overlay, 0.85, canvas[y1:y2, x1:x2], 0.15, 0, canvas[y1:y2, x1:x2])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 2)

        # Pulse circle for push-to-record
        pulse = 0.5 + 0.5 * math.sin(time.time() * 7)
        rec_col = (40, 40, int(160 + 95 * pulse)) if (key_held or is_rec) else (80, 90, 100)
        cv2.circle(canvas, (x1 + 25, y1 + 30), 8, rec_col, -1, cv2.LINE_AA)

        rec_status = "RECORDING ACTIVE - RELEASE SPACE WHEN DONE" if (key_held or is_rec) else "PUSH-TO-RECORD: HOLD SPACE WHILE MOVING"
        cv2.putText(canvas, rec_status, (x1 + 42, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 240), 1, cv2.LINE_AA)

        # Step Title or Mode Heading
        if mode == 'GUIDED':
            step_idx = hud_info.get('step_idx', 0)
            total = hud_info.get('total_steps', 7)
            title = hud_info.get('step_title', 'Calibration')
            desc = hud_info.get('step_desc', '')
            step_str = f"STEP {step_idx+1} OF {total}: {title}"
            cv2.putText(canvas, step_str, (x1 + 25, y1 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, desc, (x1 + 25, y1 + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 200, 255), 1, cv2.LINE_AA)

            # Hold progress bar for REST step
            if step_idx == 0:
                prog = hud_info.get('hold_progress', 0.0)
                bar_w = card_w - 50
                cv2.rectangle(canvas, (x1 + 25, y1 + 124), (x1 + 25 + bar_w, y1 + 144), (40, 45, 55), -1)
                cv2.rectangle(canvas, (x1 + 25, y1 + 124), (x1 + 25 + int(bar_w * prog), y1 + 144), (80, 220, 120), -1)
                pct_str = f"Noise Calibration Progress: {int(prog*100)}%"
                cv2.putText(canvas, pct_str, (x1 + 35, y1 + 139),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                peak_disp = hud_info.get('stroke_peak_disp', 0.0)
                disp_str = f"Live Peak Deflection: {peak_disp*1000:.0f}px | Completed Strokes: {hud_info.get('completed_count', 0)}"
                cv2.putText(canvas, disp_str, (x1 + 25, y1 + 138),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 100), 1, cv2.LINE_AA)

        elif mode == 'FREEFORM':
            cv2.putText(canvas, "NATURAL SCROLLING SESSION", (x1 + 25, y1 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Scroll / flick freely as you naturally would. Hold SPACE during any gesture to log it.", (x1 + 25, y1 + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 200, 255), 1, cv2.LINE_AA)
            logged = hud_info.get('completed_count', 0)
            cv2.putText(canvas, f"Logged Gestures: {logged} | Press 'A' to analyze, 'Q' to save", (x1 + 25, y1 + 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 100), 1, cv2.LINE_AA)

        elif mode == 'DRILL':
            d_dir = hud_info.get('drill_target_direction', 'UP')
            d_cnt = hud_info.get('drill_count', 0)
            d_tot = hud_info.get('drill_total', 5)
            cv2.putText(canvas, f"REPETITION DRILL: {d_dir} FLICKS ({d_cnt}/{d_tot})", (x1 + 25, y1 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Perform 5 deliberate {d_dir} flicks in a row. Hold SPACE while moving.", (x1 + 25, y1 + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Current Count: {d_cnt} of {d_tot}", (x1 + 25, y1 + 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 100), 1, cv2.LINE_AA)

    def _draw_oscilloscope(self, canvas: np.ndarray, osc_samples: List[Dict[str, Any]]):
        """Draws real-time scrolling displacement and velocity waveforms."""
        h, w = canvas.shape[:2]
        osc_x = 20
        osc_w = w - 40
        osc_y = h - 145
        osc_h = 85

        # Background
        cv2.rectangle(canvas, (osc_x, osc_y), (osc_x + osc_w, osc_y + osc_h), (18, 22, 28), -1)
        cv2.rectangle(canvas, (osc_x, osc_y), (osc_x + osc_w, osc_y + osc_h), (45, 55, 70), 1)

        # Baseline zero line
        mid_y = osc_y + osc_h // 2
        cv2.line(canvas, (osc_x, mid_y), (osc_x + osc_w, mid_y), (60, 75, 95), 1)

        # Dotted Threshold Lines (+/- 20px)
        cv2.line(canvas, (osc_x, mid_y - 22), (osc_x + osc_w, mid_y - 22), (40, 120, 80), 1)
        cv2.line(canvas, (osc_x, mid_y + 22), (osc_x + osc_w, mid_y + 22), (40, 120, 80), 1)
        cv2.putText(canvas, "+THRESHOLD (UP)", (osc_x + 8, mid_y - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 180, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "-THRESHOLD (DOWN)", (osc_x + 8, mid_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 180, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LIVE OSCILLOSCOPE (DISPLACEMENT y(t))", (osc_x + osc_w - 240, osc_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 150), 1, cv2.LINE_AA)

        if not osc_samples or len(osc_samples) < 2:
            return

        # Plot samples
        n = len(osc_samples)
        dx = osc_w / max(1, n - 1)
        for i in range(1, n):
            x1 = int(osc_x + (i - 1) * dx)
            x2 = int(osc_x + i * dx)
            # Normalized displacement: 0.05 is roughly 40 pixels on osc
            disp1 = osc_samples[i - 1]['disp']
            disp2 = osc_samples[i]['disp']
            y1 = int(mid_y - (disp1 * 700.0))
            y2 = int(mid_y - (disp2 * 700.0))
            y1 = max(osc_y + 2, min(osc_y + osc_h - 2, y1))
            y2 = max(osc_y + 2, min(osc_y + osc_h - 2, y2))

            # Color: Cyan if active stroke, Green if stable
            col = (80, 240, 255) if osc_samples[i]['is_moving'] else (80, 180, 100)
            cv2.line(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        y = h - 45
        cv2.rectangle(canvas, (0, y), (w, h), (18, 20, 24), -1)
        cv2.line(canvas, (0, y), (w, y), (45, 50, 60), 1)

        controls = "SPACE: Push-To-Record  |  1: Guided  |  2: Freeform  |  3: Drill  |  V: Camera Feed  |  A: Analyze  |  Q: Save & Quit"
        cv2.putText(canvas, controls, (20, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    logger.info("🎬 Initializing HandScroller Training Station...")

    cap = setup_camera(args.camera)
    cap = LatestFrameCamera(cap)
    gesture_engine = GestureEngine()

    engine = TrainingStationEngine()
    ui = TrainingStationUI()

    mode_map = {'guided': engine.MODE_GUIDED, 'freeform': engine.MODE_FREEFORM, 'drill': engine.MODE_DRILL}
    engine.start_session(mode=mode_map.get(args.mode, engine.MODE_GUIDED))

    show_camera_feed = args.show_camera or SHOW_CAMERA_FEED
    mirror = MIRROR_MODE
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)

    print("\n" + "=" * 70)
    print("🖐️  HANDSCROLLER TRAINING STATION IS READY")
    print("=" * 70)
    print("Hold SPACE while performing each flick/gesture, then release SPACE.")
    print("Keys: [1] Guided, [2] Freeform, [3] Drill, [V] Privacy/Video, [A] Analyze, [Q] Quit\n")

    running = True
    try:
        while running:
            ret, frame = cap.read()
            if ret is None:
                kp = cv2.pollKey()
                if kp != -1 and (kp & 0xFF) in (ord('q'), 27):
                    break
                continue
            if not ret:
                logger.error("Camera disconnected.")
                break

            now = cap.timestamp or time.monotonic()
            if mirror:
                frame = cv2.flip(frame, 1)

            # Convert RGB for MediaPipe inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _ = gesture_engine.process_frame(rgb_frame, now=now)
            landmarks = gesture_engine.get_current_landmarks()

            # Process training station engine update
            hud_info = engine.update(landmarks, now=now)

            # Render UI
            display_frame = frame.copy() if show_camera_feed else np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            if display_frame.shape[0] != WIN_H or display_frame.shape[1] != WIN_W:
                display_frame = cv2.resize(display_frame, (WIN_W, WIN_H))

            ui.render(display_frame, landmarks, hud_info, privacy_mode=not show_camera_feed)
            cv2.imshow(WINDOW_NAME, display_frame)

            # Keyboard handling
            k = cv2.waitKey(1)
            key = (k & 0xFF) if k != -1 else -1
            while True:
                kp = cv2.pollKey()
                if kp == -1:
                    break
                if key == -1:
                    key = kp & 0xFF

            if key in (ord('q'), 27):  # Q or ESC
                if engine.active:
                    engine._finalize_and_save_session()
                logger.info("👋 Training Station completed. Exiting.")
                break

            elif key == ord('1'):
                logger.info("Switching to Mode: GUIDED")
                engine.start_session(engine.MODE_GUIDED)

            elif key == ord('2'):
                logger.info("Switching to Mode: FREEFORM")
                engine.start_session(engine.MODE_FREEFORM)

            elif key == ord('3'):
                logger.info("Switching to Mode: DRILL")
                engine.start_session(engine.MODE_DRILL)

            elif key == ord('v'):
                show_camera_feed = not show_camera_feed
                logger.info("📷 Camera view: %s", "VISIBLE" if show_camera_feed else "HIDDEN (Privacy Canvas)")

            elif key == ord('m'):
                mirror = not mirror
                logger.info("🪞 Mirror Mode: %s", "ON" if mirror else "OFF")

            elif key == ord('a'):
                if engine.active:
                    engine._finalize_and_save_session()
                if engine.latest_profile:
                    print("\n" + "=" * 60)
                    print(KinematicsAnalyzer.generate_markdown_report(engine.latest_profile))
                    print("=" * 60 + "\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
