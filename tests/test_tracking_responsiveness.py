"""Full-view acquisition, fresh capture, and eased input regression coverage."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.camera_capture import LatestFrameCamera
from app.gesture_engine import GestureEngine, GestureResult
from app.input_simulator import InputSimulator
from app.intent_tracker import RelaxedIntentTracker
from app.learning_curve import LearningCurve, UserProfile, load_profile
from app.low_light import LowLightEnhancer
from app.smooth_scroll import SmoothWheel
from test_relaxed_scrolling import Replay, make_hand


@__import__('unittest.mock').mock.patch('app.gesture_engine.MOTION_GESTURES_ENABLED', True)
class AcquisitionTests(unittest.TestCase):
    def test_ready_in_under_150ms_anywhere_in_view(self):
        for x in (-0.33, 0, 0.30):
            for y in (-0.40, 0):
                with self.subTest(x=x, y=y):
                    tracker = RelaxedIntentTracker()
                    for i in range(5):
                        self.assertIsNone(tracker.update(make_hand(hand_x=x, hand_y=y), 1000 + i / 30))
                    self.assertEqual(tracker.state, 'READY')
                    self.assertLessEqual(tracker.baseline.time - 1000, 0.15)

    def test_relaxed_moving_finger_does_not_block_acquisition(self):
        tracker = RelaxedIntentTracker()
        for i in range(5):
            self.assertIsNone(tracker.update(make_hand(fingers={12: (0, 0.02 * (i % 2))}), 1000 + i / 30))
        self.assertIsNotNone(tracker.baseline)
        self.assertLessEqual(tracker.baseline.time - 1000, 0.15)

    def test_vertical_stroke_after_sideways_reposition(self):
        for x in (-0.32, 0.30):
            replay = Replay().feed(0.3)
            replay.feed(0.4, hand_x=lambda t: x * t).feed(0.2, hand_x=x)
            self.assertEqual(replay.events, [])
            replay.feed(0.3, hand_x=x, hand_y=lambda t: -0.035 * t)
            replay.feed(0.3, hand_x=x, hand_y=lambda t: -0.035 * (1 - t)).feed(0.6, hand_x=x)
            self.assertEqual(replay.types(), ['SWIPE_UP'])

    def test_decisive_stroke_acts_before_hand_stops(self):
        replay = Replay().feed(0.3)
        replay.feed(0.23, dy=lambda t: -0.035 * t)
        self.assertEqual(replay.types(), ['SWIPE_UP'])
        replay.feed(0.3, dy=lambda t: -0.035 * (1 - t)).feed(0.6)
        self.assertEqual(replay.types(), ['SWIPE_UP'])

    def test_corrupt_learned_thresholds_cannot_block_small_strokes(self):
        replay = Replay().feed(0.3)
        replay.tracker.set_learned_thresholds(2.3, 3.2)
        replay.feed(0.3).feed(0.4, dy=lambda t: -0.014 * t).feed(0.2, dy=-0.014)
        self.assertEqual(replay.types(), ['SWIPE_UP'])

    def test_engine_receives_entire_frame_at_all_positions(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in (-0.33, 0.30):
            def detect(image, timestamp):
                self.assertEqual((image.height, image.width), (480, 640))
                return SimpleNamespace(hand_landmarks=[make_hand(hand_x=x)])
            fake = SimpleNamespace(detect_for_video=detect, close=lambda: None)
            engine = GestureEngine(fake)
            for i in range(5):
                engine.process_frame(frame, now=1000 + i / 30)
            self.assertTrue(engine.get_hand_info()['tracking'])
            self.assertEqual(engine.hand_state.intent.state, 'READY')
            engine.cleanup()


class ProfileTests(unittest.TestCase):
    def test_large_legacy_thresholds_load_as_usable_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'profile.json'
            p.write_text(json.dumps(dict(calibrated=True, finger_distance=2.3, hand_distance=3.2, phone_style=False)))
            with patch('app.learning_curve.PROFILE_PATH', str(p)):
                profile = load_profile()
                self.assertEqual(profile.finger_distance, 0.038)
                self.assertEqual(profile.hand_distance, 0.055)
                self.assertFalse(profile.phone_style)
                self.assertTrue(profile.calibrated)

    def test_changed_rest_postures_are_not_sensor_noise(self):
        with patch('app.learning_curve.load_profile', return_value=UserProfile()):
            learning = LearningCurve(phase_duration=1)
        learning.start()
        for i in range(31):
            learning.update(make_hand(hand_y=-0.10 if i < 10 else (0.10 if i < 20 else 0)), 1000 + i / 30)
        self.assertEqual(learning.state, learning.STATE_DOWN)
        self.assertLessEqual(learning.relax_noise_floor, 0.020)
        learning.relax_noise_floor = 1.3
        learning.profile.down_stroke_recorded = 3
        learning.profile.up_stroke_recorded = 2
        with patch('app.learning_curve.save_profile'), patch('app.learning_curve.open_test_page'):
            learning._finalize_calibration(1001)
        self.assertLessEqual(learning.profile.finger_distance, 0.065)
        self.assertLessEqual(learning.profile.hand_distance, 0.085)


class LightingTests(unittest.TestCase):
    def test_dim_input_is_actually_measured_and_brightened(self):
        enhancer = LowLightEnhancer()
        frame = np.full((120, 160, 3), 35, dtype=np.uint8)
        enhanced = enhancer.prepare(frame)
        self.assertGreater(enhanced.mean(), frame.mean())
        self.assertGreater(enhancer.get_brightness_level()[1], 0)

    def test_bright_input_skips_expensive_enhancement(self):
        enhancer = LowLightEnhancer()
        frame = np.full((120, 160, 3), 160, dtype=np.uint8)
        with patch.object(enhancer, 'enhance_fast') as enhance:
            self.assertIs(enhancer.prepare(frame), frame)
            enhance.assert_not_called()


class SmoothWheelTests(unittest.TestCase):
    def test_exact_distance_and_direction_at_varied_frame_rates(self):
        for fps in (6, 15, 30, 60):
            for sign in (-1, 1):
                deltas = []
                wheel = SmoothWheel(lambda value: deltas.append(value) or True, lambda: 123)
                self.assertTrue(wheel.start(sign * 960, now=1000))
                for i in range(1, fps + 1):
                    wheel.update(1000 + i / fps)
                self.assertEqual(sum(deltas), sign * 960)
                self.assertGreater(len(deltas), 1)
                self.assertTrue(all(delta * sign > 0 for delta in deltas))
                self.assertFalse(wheel.active)

    def test_focus_change_cancels_remaining_input(self):
        target = [123]
        deltas = []
        wheel = SmoothWheel(lambda value: deltas.append(value) or True, lambda: target[0])
        wheel.start(-960, now=1000)
        target[0] = 456
        self.assertFalse(wheel.update(1000.1))
        self.assertEqual(deltas, [-120])
        self.assertFalse(wheel.active)

    def test_send_failure_stops_animation(self):
        outcomes = iter((True, False))
        wheel = SmoothWheel(lambda value: next(outcomes), lambda: 123)
        self.assertTrue(wheel.start(960, now=1000))
        self.assertFalse(wheel.update(1000.1))
        self.assertEqual(wheel.sent, 120)
        self.assertFalse(wheel.active)

    def test_pause_cancels_pending_and_undo_uses_actual_partial_distance(self):
        deltas = []
        with patch('app.input_simulator._get_foreground_handle', return_value=123), patch('app.input_simulator._get_foreground_window_title', return_value='Browser'), patch('app.input_simulator._ensure_cursor_over_target'), patch.object(InputSimulator, '_send_wheel_delta', side_effect=lambda value: deltas.append(value) or True):
            sim = InputSimulator('scroll')
            action = sim.execute_gesture(GestureResult('SWIPE_UP'))
            sim.update_scroll(sim._wheel.started + 0.06)
            sent = sum(deltas)
            self.assertGreater(sent, 0)
            self.assertLess(sent, 960)
            sim.pause()
            sim.update_scroll(sim._wheel.started + 1)
            self.assertEqual(sum(deltas), sent)
            self.assertTrue(sim.undo_action(action))
            self.assertEqual(sum(deltas), 0)


class CameraCaptureTests(unittest.TestCase):
    def test_backlog_is_replaced_and_shutdown_releases_once(self):
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
        release_count = []
        finished = threading.Event()
        def read():
            if frames:
                return True, frames.pop(0)
            finished.set()
            return False, None
        camera = LatestFrameCamera(SimpleNamespace(read=read, release=lambda: release_count.append(1)))
        try:
            self.assertTrue(finished.wait(1))
            ok, frame = camera.read()
            self.assertTrue(ok)
            self.assertEqual(int(frame[0, 0, 0]), 2)
            self.assertIsNotNone(camera.timestamp)
            self.assertIsNone(camera.read(timeout=0)[0])
        finally:
            camera.release()
        self.assertEqual(release_count, [1])

    def test_dead_camera_reports_end_without_replaying_stale_frames(self):
        camera = LatestFrameCamera(SimpleNamespace(read=lambda: (False, None), release=lambda: None))
        camera._thread.join(1)
        self.assertEqual(camera.read(), (False, None))
        camera.release()


class CameraExposureTests(unittest.TestCase):
    def make_capture(self, backend):
        from unittest.mock import Mock
        import cv2
        capture = Mock()
        capture.get.side_effect = lambda prop: backend if prop == cv2.CAP_PROP_BACKEND else -4
        capture.set.return_value = True
        return capture

    def test_msmf_auto_keeps_valid_exposure_instead_of_fractional_flag(self):
        import cv2
        from main import configure_camera_exposure
        capture = self.make_capture(cv2.CAP_MSMF)
        with patch('main.CAMERA_AUTO_EXPOSURE', 1):
            self.assertEqual(configure_camera_exposure(capture), (True, True))
        capture.set.assert_called_once_with(cv2.CAP_PROP_AUTO_EXPOSURE, -4)

    def test_directshow_auto_uses_boolean_flag(self):
        import cv2
        from main import configure_camera_exposure
        capture = self.make_capture(cv2.CAP_DSHOW)
        with patch('main.CAMERA_AUTO_EXPOSURE', 1):
            self.assertEqual(configure_camera_exposure(capture), (True, True))
        capture.set.assert_called_once_with(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)

    def test_manual_mode_sets_a_real_exposure_after_disabling_auto(self):
        import cv2
        from main import configure_camera_exposure
        capture = self.make_capture(cv2.CAP_MSMF)
        with patch('main.CAMERA_AUTO_EXPOSURE', 0), patch('main.CAMERA_EXPOSURE', -6):
            self.assertEqual(configure_camera_exposure(capture), (False, True))
        self.assertEqual(capture.set.call_args_list[-1].args, (cv2.CAP_PROP_EXPOSURE, -6))


if __name__ == '__main__':
    unittest.main()
