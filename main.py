"""
╔══════════════════════════════════════════════════════════════════╗
║                    🖐️  HandScroller  v1.0                       ║
║         Hands-Free Scrolling for TikTok, Reels & Shorts         ║
║                                                                  ║
║  Swipe your hand in front of the camera to scroll content.       ║
║  Small relaxed finger and whole-hand strokes.                   ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python main.py [--no-preview] [--mode scroll|keyboard|smart] [--sensitivity 1.0]

Controls (when preview window is active):
    Q / ESC  →  Quit
    R        →  Record 6 Reference Positions (3 UP, 3 DOWN)
    SPACE    →  Lock in current posture immediately while recording
    P        →  Pause / Resume gesture detection
    M        →  Toggle mirror mode
    +/-      →  Adjust sensitivity
    S        →  Cycle scroll mode (scroll → keyboard → smart)
    H        →  Hide preview (Ctrl+C in console to stop when hidden)
    F        →  Toggle fast/quality low-light enhancement
    C        →  Recalibrate your relaxed hand position
    D        →  Toggle phone/natural scroll direction
"""

import os
import sys

# Ensure HandScroller root is in sys.path and is current working directory
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
import cv2
import time
import logging
import argparse
import math
import numpy as np

from app.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CAMERA_EXPOSURE, CAMERA_AUTO_EXPOSURE, CAMERA_BACKEND,
    LOW_LIGHT_ENABLED, SHOW_CAMERA_PREVIEW, SHOW_CAMERA_FEED,
    WINDOW_WIDTH, WINDOW_HEIGHT, MIRROR_MODE,
    WINDOW_NAME, SCROLL_MODE, PHONE_STYLE_SCROLL, SCROLL_AMOUNT,
)
from app.gesture_engine import GestureEngine
from app.low_light import LowLightEnhancer
from app.input_simulator import InputSimulator
from app.ui_overlay import UIOverlay
from app.learning_curve import LearningCurve, load_profile, save_profile, open_test_page
from app.gesture_recorder import GestureRecorder, load_references
from app.adaptive_learner import AdaptiveLearner
from app.camera_capture import LatestFrameCamera

# ─── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("HandScroller")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HandScroller - Hands-free scrolling with hand gestures"
    )
    parser.add_argument(
        '--no-preview', action='store_true',
        help='Run without camera preview window (headless mode)'
    )
    parser.add_argument(
        '--show-camera', action='store_true',
        help='Show actual webcam video (disabled by default for privacy)'
    )
    parser.add_argument(
        '--mode', choices=['scroll', 'keyboard', 'smart'],
        default=SCROLL_MODE,
        help='Scroll mode: scroll (mouse wheel), keyboard (arrow keys), smart (auto-detect)'
    )
    parser.add_argument(
        '--sensitivity', type=float, default=None,
        help='Gesture sensitivity (0.5=less sensitive, 2.0=very sensitive)'
    )
    parser.add_argument(
        '--camera', type=int, default=CAMERA_INDEX,
        help='Camera device index (0=default webcam)'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Use fast low-light enhancement (default; F toggles quality mode)'
    )
    parser.add_argument('--direction', choices=['phone', 'natural'],
                        default=None,
                        help='phone: hand up scrolls down; natural: hand up scrolls up')
    parser.add_argument('--scroll-amount', type=int, default=SCROLL_AMOUNT,
                        help='Mouse wheel steps per gesture (1-20)')
    parser.add_argument('--diagnostic-seconds', type=float, default=0,
                        help='Run camera/tracking checks for N seconds without sending input')
    parser.add_argument('--record', action='store_true',
                        help='Launch the 6-position reference recorder (3 UP, 3 DOWN positions)')
    parser.add_argument('--train', action='store_true',
                        help='Launch the Training Station to record hand movements and tune parameters')
    parser.add_argument('--learn', action='store_true',
                        help='Launch the interactive Learning Curve calibration wizard')
    parser.add_argument('--gesture-undo', action='store_true',
                        help='Enable sideways-wag undo (off by default; never changes settings)')
    parser.add_argument('--motion', action='store_true',
                        help='Enable spatial hand motion/swipe gestures in addition to touch')
    parser.add_argument('--no-open-test', action='store_true',
                        help='Compatibility option; startup no longer opens a test page (use O)')
    args = parser.parse_args()
    if args.sensitivity is not None and (not math.isfinite(args.sensitivity) or not 0.3 <= args.sensitivity <= 3):
        parser.error('--sensitivity must be between 0.3 and 3.0')
    if not 1 <= args.scroll_amount <= 20:
        parser.error('--scroll-amount must be between 1 and 20')
    if not math.isfinite(args.diagnostic_seconds) or args.diagnostic_seconds < 0:
        parser.error('--diagnostic-seconds must be a finite nonnegative number')
    return args


def apply_control_preferences(gesture_engine, input_sim, profile, args):
    """Explicit command-line preferences take precedence over saved calibration."""
    gesture_engine.apply_profile(profile)
    if getattr(args, 'motion', False):
        import app.gesture_engine as ge_mod
        import app.config as cfg
        ge_mod.MOTION_GESTURES_ENABLED = True
        cfg.MOTION_GESTURES_ENABLED = True
        logger.info("🖐️ Hand Motion Swipes: ENABLED via --motion")
    if args.direction is not None:
        input_sim.phone_style = args.direction == 'phone'
    else:
        input_sim.phone_style = profile.phone_style
    if args.sensitivity is not None:
        gesture_engine.set_sensitivity(args.sensitivity)


def remember_controls(gesture_engine, input_sim):
    """Persist controls only when the user explicitly changes them."""
    profile = load_profile()
    profile.phone_style = input_sim.phone_style
    profile.sensitivity = gesture_engine.sensitivity
    return save_profile(profile)


def update_scroll_feedback(input_sim, overlay):
    """Surface late delivery failures once, even between camera frames."""
    input_sim.update_scroll()
    action = input_sim.pop_scroll_error()
    if action is not None:
        logger.warning('%s: %s', action['description'], action['error'])
        overlay.trigger_gesture_flash(action['gesture'], action['description'], success=False)


def configure_camera_exposure(cap):
    """Translate exposure controls for the active Windows camera backend."""
    automatic = CAMERA_AUTO_EXPOSURE not in (0, 0.25)
    backend = int(cap.get(cv2.CAP_PROP_BACKEND))
    if automatic:
        if backend == cv2.CAP_MSMF:
            # MSMF passes this value to IAMCameraControl.Set as well as using
            # nonzero to select Auto. 0.75/1 truncate to invalid exposure values
            # on many webcams; preserve the camera's valid exposure instead.
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            value = exposure if math.isfinite(exposure) and exposure < 0 else CAMERA_EXPOSURE
        elif backend == cv2.CAP_DSHOW:
            value = 1.0
        else:
            value = CAMERA_AUTO_EXPOSURE
        accepted = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0 if backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW) else 0.25)
        accepted = cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    if not accepted:
        logger.warning('Camera driver rejected %s exposure; keeping its current setting.',
                       'automatic' if automatic else 'manual')
    return automatic, accepted


def setup_camera(camera_index=0):
    """
    Initialize camera with settings optimized for hand tracking.
    Keeps useful exposure and image detail for stable tracking.
    """
    logger.info(f"Opening camera {camera_index}...")

    # Determine backends to attempt
    if CAMERA_BACKEND == "DSHOW":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif CAMERA_BACKEND == "MSMF":
        backends = [cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]
    else:  # "AUTO" or default: prioritize Media Foundation for smooth 30 FPS
        backends = [cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]

    cap = None
    for backend in backends:
        try:
            temp_cap = cv2.VideoCapture(camera_index, backend)
            if temp_cap.isOpened():
                cap = temp_cap
                break
            temp_cap.release()
        except Exception:
            pass

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        if cap:
            cap.release()
        logger.error("❌ Could not open camera! Check if another app is using it.")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # Let the camera expose the hand properly; software brightening cannot
    # recover detail lost by forcing a dark exposure.
    automatic_exposure, exposure_accepted = configure_camera_exposure(cap)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Log actual camera settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"📷 Camera ready: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    if exposure_accepted:
        logger.info('Camera exposure: %s', 'automatic' if automatic_exposure else 'manual')

    return cap


def print_banner():
    """Print the startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║              🖐️  HandScroller  v1.0                  ║
    ║        Hands-Free Social Media Scrolling             ║
    ╠══════════════════════════════════════════════════════╣
    ║                                                      ║
    ║  Touch Thumb + Index  →  Scroll DOWN ⬇️              ║
    ║  Touch Thumb + Middle →  Scroll UP   ⬆️              ║
    ║  Hand UP / Hand DOWN  →  Fluid Swipe Scroll          ║
    ║                                                      ║
    ║  S: Cycle Mode (Keyboard/Scroll) D: Reverse Direction║
    ║  G: Toggle Swipes (Motion/Touch) V: View camera      ║
    ║  C: Quick Calibrate   P: Pause   🔒 Privacy Mode     ║
    ║  O: Open interactive test page   +/-: Sensitivity    ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def disable_quickedit():
    """Disable QuickEdit mode in Windows Console so clicking or holding keys never pauses execution."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE = -10
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            mode.value &= ~0x0040  # ENABLE_QUICK_EDIT_MODE
            mode.value |= 0x0080   # ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(handle, mode.value)
    except Exception:
        pass


def main():
    """Main application loop."""
    disable_quickedit()
    args = parse_args()
    print_banner()

    # ─── Initialize Components ────────────────────────────────
    cap = setup_camera(args.camera)
    try:
        gesture_engine = GestureEngine(gesture_undo=args.gesture_undo)
        low_light = LowLightEnhancer()
        input_sim = InputSimulator(mode=args.mode)
        input_sim.set_scroll_amount(args.scroll_amount)
        overlay = UIOverlay()
    except Exception:
        cap.release()
        raise

    # Apply control preferences
    profile = load_profile()
    apply_control_preferences(gesture_engine, input_sim, profile, args)

    # Apply settings
    show_preview = SHOW_CAMERA_PREVIEW and not args.no_preview and not args.diagnostic_seconds
    show_camera_feed = args.show_camera or SHOW_CAMERA_FEED
    mirror = MIRROR_MODE
    use_fast_enhance = True  # Fast enhancement is the default for responsive tracking.

    if show_preview:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    logger.info(f"🎮 Mode: {input_sim.mode} | Sensitivity: {gesture_engine.sensitivity:.1f}")
    direction_desc = 'natural (hand UP = scroll UP ⬆️)' if not input_sim.phone_style else 'phone (hand UP = scroll DOWN / Next ⬇️)'
    logger.info('Direction: %s (Press D to toggle)', direction_desc)
    logger.info(f"🔒 Privacy Mode: {'ACTIVE (You are hidden)' if not show_camera_feed else 'OFF (Camera video visible)'}")
    logger.info(f"🖥️  Preview HUD: {'ON' if show_preview else 'OFF'} | Mirror: {'ON' if mirror else 'OFF'}")
    logger.info("⚡ Pure deterministic code engine active: 1ms instant hand detection, zero virtual boxes")
    logger.info("👉 Touch Thumb + Index -> Scroll DOWN ⬇️ | Touch Thumb + Middle -> Scroll UP ⬆️")
    logger.info("👉 Fluid hand/finger strokes also supported | S: Cycle Mode | D: Toggle Direction")
    if args.diagnostic_seconds:
        logger.info('Diagnostic mode: observing tracking for %.1fs, input disabled', args.diagnostic_seconds)

    logger.info("Press Q or ESC to quit.\n")

    # Drain the camera continuously; process only its newest full-view frame.
    cap = LatestFrameCamera(cap)

    # ─── 6-Position Reference Recorder Setup ──────────────────
    def on_record_complete(refs):
        gesture_engine.load_references(refs)
        logger.info("🎉 6-Position References successfully locked & active in tracking engine!")
        overlay.trigger_gesture_flash("PALM_HOLD", "References Saved & Active!", success=True)
        open_test_page()

    gesture_recorder = GestureRecorder(on_complete_callback=on_record_complete)
    if args.record:
        gesture_recorder.start()

    # ─── Main Loop ────────────────────────────────────────────
    running = True
    preview_visible = show_preview
    last_gesture_log_time = 0
    started = time.monotonic()
    diagnostic_states = {}

    try:
        while running:
            if args.diagnostic_seconds and time.monotonic() - started >= args.diagnostic_seconds:
                break
            update_scroll_feedback(input_sim, overlay)
            ret, frame = cap.read()
            if ret is None:
                # Keep wheel easing and quit controls responsive between camera frames.
                update_scroll_feedback(input_sim, overlay)
                kp = cv2.pollKey()
                if kp != -1 and (kp & 0xFF) in (ord('q'), 27):
                    break
                continue
            if not ret:
                logger.error('Camera stopped returning frames. Close other camera apps and restart.')
                break
            frame_time = cap.timestamp

            # Mirror if enabled (more natural feel)
            if mirror:
                frame = cv2.flip(frame, 1)

            # ─── Frame Preparation for Tracking ───────────────
            # Keep full 30 FPS speed; adaptively enhance only when lighting is actually dim
            if LOW_LIGHT_ENABLED:
                feed_frame = low_light.prepare(frame, fast=use_fast_enhance)
            else:
                feed_frame = frame

            # Convert to RGB for MediaPipe (natural RGB ensures optimal hand detection)
            rgb_frame = cv2.cvtColor(feed_frame, cv2.COLOR_BGR2RGB)

            # ─── Gesture Detection ────────────────────────────
            motion_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if feed_frame is not frame else rgb_frame
            gesture = gesture_engine.process_frame(
                rgb_frame, now=frame_time, motion_frame_rgb=motion_rgb)
            update_scroll_feedback(input_sim, overlay)
            hand_info = gesture_engine.get_hand_info()
            landmarks = gesture_engine.get_current_landmarks()
            brightness_info = low_light.get_brightness_level()
            if args.diagnostic_seconds:
                status = hand_info['intent_status']
                diagnostic_states[status] = diagnostic_states.get(status, 0) + 1

            # ─── 6-Position Reference Recorder Update ─────────
            if gesture_recorder.is_active():
                is_key_held = gesture_recorder.check_physical_key_held()
                recorder_info = gesture_recorder.update(landmarks, frame_rgb=rgb_frame, now=frame_time, key_held=is_key_held)
                gesture = None  # Suppress scrolling inputs during reference calibration
            else:
                recorder_info = None

            # ─── Execute Gesture Action ───────────────────────
            if gesture:
                action = ({'description': f'Detected {gesture.type} ({gesture.source})', 'mode': 'diagnostic', 'success': True}
                          if args.diagnostic_seconds else input_sim.execute_gesture(gesture))

                # Auto-switch to browser if gesture was blocked because preview HUD or terminal had focus
                if not args.diagnostic_seconds and not action.get('success'):
                    err = action.get('error', '')
                    if err in ('HandScroller is focused', 'Terminal is focused', 'No target application is focused'):
                        from app.window_manager import ensure_target_active
                        if ensure_target_active():
                            action = input_sim.execute_gesture(gesture)

                # Log gesture (rate-limited)
                now = time.time()
                if now - last_gesture_log_time > 0.3:
                    desc = action.get('description', gesture.type)
                    mode = action.get('mode', 'unknown')
                    log = logger.info if action.get('success') else logger.warning
                    log('%s (mode=%s, source=%s, quality=%.2f)', desc, mode, gesture.source, gesture.confidence)
                    last_gesture_log_time = now

                # Trigger visual feedback
                overlay.trigger_gesture_flash(
                    gesture.type,
                    action.get('description', gesture.type), success=action.get('success', False)
                )

            # ─── Draw Preview ─────────────────────────────────
            if preview_visible:
                if show_camera_feed:
                    display_frame = frame.copy()
                else:
                    # Privacy Canvas: Clean dark background - camera feed of user is completely hidden!
                    display_frame = np.full_like(frame, 18)

                input_stats = input_sim.get_stats()
                overlay.is_paused = not gesture_engine.is_active()
                overlay.draw(
                    display_frame, landmarks, hand_info,
                    brightness_info, input_stats,
                    privacy_mode=not show_camera_feed,
                    recorder_info=recorder_info
                )

                cv2.imshow(WINDOW_NAME, display_frame)

            # ─── Handle Keyboard Input & Drain Key-Repeat Queue ──
            # When holding keys (e.g. SPACE to record), Windows sends 30-50 key-repeat events/sec.
            # Draining all queued events prevents HighGUI queue saturation and eliminates video freeze.
            k = cv2.waitKey(1)
            key = (k & 0xFF) if k != -1 else -1
            while True:
                kp = cv2.pollKey()
                if kp == -1:
                    break
                if key == -1:
                    key = kp & 0xFF

            if key == ord('q') or (key == 27 and not gesture_recorder.is_active()):  # Q or ESC
                logger.info("👋 Quitting...")
                running = False

            elif key == 27 and gesture_recorder.is_active():  # ESC cancels recording
                gesture_recorder.cancel()
                logger.info("❌ Reference recording cancelled")

            elif key in (32, 13):  # SPACE or ENTER
                # Push-to-record is handled directly and reliably via physical key hold & release in gesture_recorder.update()
                pass

            elif key == ord('r'):  # R starts 6-position reference recorder
                logger.info("🎬 Starting 6-Position Reference Recorder (3 UP, 3 DOWN)...")
                gesture_recorder.start()

            elif key == ord('p'):  # Pause/Resume
                if gesture_engine.is_active():
                    gesture_engine.pause()
                    input_sim.pause()
                    logger.info("⏸️  Paused")
                else:
                    gesture_engine.resume()
                    input_sim.resume()
                    logger.info("▶️  Resumed")

            elif key == ord('v'):  # Toggle camera feed (Privacy Mode)
                show_camera_feed = not show_camera_feed
                logger.info(f"📷 Camera Video: {'VISIBLE' if show_camera_feed else 'HIDDEN (Privacy Mode)'}")

            elif key == ord('m'):  # Toggle mirror
                mirror = not mirror
                gesture_engine.recalibrate()
                logger.info(f"🪞 Mirror: {'ON' if mirror else 'OFF'}")

            elif key == ord('c'):
                gesture_engine.recalibrate()
                logger.info('Recalibrating: relax your hand in camera view')

            elif key == ord('d'):
                input_sim.cancel_scroll()
                input_sim.phone_style = not input_sim.phone_style
                remember_controls(gesture_engine, input_sim)
                gesture_engine.recalibrate()
                logger.info('Direction: %s', 'phone (hand UP = scroll DOWN)' if input_sim.phone_style else 'natural (hand UP = scroll UP)')

            elif key == ord('+') or key == ord('='):  # Increase sensitivity
                new_sens = gesture_engine.sensitivity + 0.1
                gesture_engine.set_sensitivity(new_sens)
                remember_controls(gesture_engine, input_sim)
                logger.info(f"📈 Sensitivity: {gesture_engine.sensitivity:.1f}")

            elif key == ord('-') or key == ord('_'):  # Decrease sensitivity
                new_sens = gesture_engine.sensitivity - 0.1
                gesture_engine.set_sensitivity(new_sens)
                remember_controls(gesture_engine, input_sim)
                logger.info(f"📉 Sensitivity: {gesture_engine.sensitivity:.1f}")

            elif key == ord('s'):  # Cycle mode
                modes = ["keyboard", "scroll", "smart"]
                current_idx = modes.index(input_sim.mode) if input_sim.mode in modes else 0
                next_mode = modes[(current_idx + 1) % len(modes)]
                input_sim.set_mode(next_mode)
                logger.info(f"🔄 Mode: {next_mode}")

            elif key == ord('g'):  # Toggle motion gestures (hand swipes)
                import app.config as cfg
                import app.gesture_engine as ge_mod
                new_state = not ge_mod.MOTION_GESTURES_ENABLED
                ge_mod.MOTION_GESTURES_ENABLED = new_state
                cfg.MOTION_GESTURES_ENABLED = new_state
                logger.info(f"🖐️ Motion Swipes: {'ENABLED (Touch + Swipes)' if new_state else 'TOUCH ONLY'}")

            elif key == ord('l'):
                logger.info("ℹ️ HandScroller uses standardized UP/DOWN benchmark dataset from 20BN-Jester & MediaPipe.")

            elif key == ord('o'):  # Open test page
                open_test_page()

            elif key == ord('h'):  # Toggle preview HUD
                if preview_visible:
                    cv2.destroyAllWindows()
                    preview_visible = False
                    logger.info("🙈 HUD hidden (HandScroller running in background)")
                else:
                    preview_visible = True
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
                    logger.info("👁️  HUD preview shown")

            elif key == ord('f'):  # Toggle fast/quality enhancement
                use_fast_enhance = not use_fast_enhance
                logger.info(f"🌙 Enhancement: {'FAST' if use_fast_enhance else 'QUALITY'}")

    except KeyboardInterrupt:
        logger.info("\n👋 Interrupted by user")
    
    finally:
        # ─── Cleanup ─────────────────────────────────────────
        logger.info("Cleaning up...")
        
        stats = gesture_engine.get_stats()
        input_stats = input_sim.get_stats()
        
        logger.info(f"📊 Session stats:")
        logger.info(f"   Frames processed: {stats['frames_processed']}")
        logger.info(f"   Hand detections: {stats['detections']} ({stats['detection_rate']})")
        logger.info(f"   Gestures detected: {stats['gestures']}")
        logger.info(f"   Actions executed: {input_stats['total_actions']}")
        if args.diagnostic_seconds:
            logger.info('Diagnostic tracking states: %s', diagnostic_states)
        
        input_sim.cancel_scroll()
        gesture_engine.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info("✅ Done. See you next time!")


if __name__ == "__main__":
    main()
