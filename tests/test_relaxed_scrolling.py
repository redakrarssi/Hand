"""Replay real production paths without webcam access or sending desktop input."""

import ctypes
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.camera_motion import CameraMotionGuard
from app.gesture_engine import GestureEngine, GestureResult
from app.input_simulator import InputSimulator, INPUT, INPUT_MOUSE, INPUT_KEYBOARD
from app.intent_tracker import RelaxedIntentTracker


def make_hand(hand_y=0, hand_x=0, fingers=None, scale=1):
    points = [(0.50, 0.84), (0.46, 0.80), (0.42, 0.75), (0.40, 0.70), (0.39, 0.66),
              (0.46, 0.70), (0.46, 0.65), (0.46, 0.61), (0.46, 0.58),
              (0.50, 0.68), (0.50, 0.62), (0.50, 0.58), (0.50, 0.55),
              (0.54, 0.69), (0.54, 0.64), (0.54, 0.60), (0.54, 0.58),
              (0.58, 0.72), (0.58, 0.68), (0.58, 0.66), (0.58, 0.65)]
    result = [SimpleNamespace(x=0.5 + (x - 0.5) * scale + hand_x,
                              y=0.7 + (y - 0.7) * scale + hand_y, z=0)
              for x, y in points]
    for tip, (dx, dy) in (fingers or {}).items():
        result[tip].x += dx * scale
        result[tip].y += dy * scale
    return result


class Replay:
    def __init__(self, fps=30, sensitivity=1, scale=1):
        self.tracker = RelaxedIntentTracker(sensitivity)
        self.fps = fps
        self.scale = scale
        self.now = 1000.0
        self.events = []

    def feed(self, seconds, hand_y=0, hand_x=0, finger=8, dy=0, dx=0):
        count = max(1, round(seconds * self.fps))
        for i in range(count):
            fraction = (i + 1) / count
            value = lambda v: v(fraction) if callable(v) else v
            landmarks = make_hand(value(hand_y) * self.scale, value(hand_x) * self.scale,
                                  {finger: (value(dx), value(dy))}, self.scale)
            event = self.tracker.update(landmarks, self.now)
            if event:
                self.events.append(event)
            self.now += 1 / self.fps
        return self

    def types(self):
        return [event.type for event in self.events]


class IntentTests(unittest.TestCase):
    def test_small_slow_finger_strokes_at_different_frame_rates(self):
        for fps in (6, 8, 12, 15, 30, 60):
            for sign in (-1, 1):
                with self.subTest(fps=fps, sign=sign):
                    replay = Replay(fps).feed(0.65)
                    replay.feed(0.8, dy=lambda t: sign * 0.014 * t)
                    replay.feed(0.25, dy=sign * 0.014)
                    replay.feed(0.45, dy=lambda t: sign * 0.014 * (1 - t)).feed(0.8)
                    self.assertEqual(replay.types(), ['SWIPE_UP' if sign < 0 else 'SWIPE_DOWN'])

    def test_full_hand_in_both_directions(self):
        for sign in (-1, 1):
            replay = Replay().feed(0.5)
            replay.feed(0.4, hand_y=lambda t: sign * 0.035 * t)
            replay.feed(0.3, hand_y=sign * 0.035)
            replay.feed(0.45, hand_y=lambda t: sign * 0.035 * (1 - t)).feed(0.8)
            self.assertEqual(replay.types(), ['SWIPE_UP' if sign < 0 else 'SWIPE_DOWN'])
            self.assertEqual(replay.events[0].source, 'hand')

    def test_each_finger_without_open_palm_requirement(self):
        for finger in (4, 8, 12, 16, 20):
            replay = Replay().feed(0.5)
            replay.feed(0.25, finger=finger, dy=lambda t: -0.035 * t)
            replay.feed(0.25, finger=finger, dy=lambda t: -0.035 * (1 - t)).feed(0.4)
            self.assertEqual(replay.types(), ['SWIPE_UP'])

    def test_preparatory_dip_and_full_return(self):
        for sign in (-1, 1):
            replay = Replay().feed(0.5)
            replay.feed(0.12, dy=lambda t: -sign * 0.006 * t)
            replay.feed(0.3, dy=lambda t: -sign * 0.006 + sign * 0.041 * t)
            replay.feed(0.3, dy=lambda t: sign * 0.035 * (1 - t)).feed(0.5)
            self.assertEqual(replay.types(), ['SWIPE_UP' if sign < 0 else 'SWIPE_DOWN'])

    def test_repeat_strokes_and_intentional_reverse(self):
        replay = Replay().feed(0.5)
        for sign in (-1, -1, 1, 1, -1):
            replay.feed(0.3, dy=lambda t: sign * 0.035 * t)
            replay.feed(0.3, dy=lambda t: sign * 0.035 * (1 - t)).feed(0.85)
        self.assertEqual(replay.types(), ['SWIPE_UP', 'SWIPE_UP', 'SWIPE_DOWN', 'SWIPE_DOWN', 'SWIPE_UP'])

    def test_delayed_slow_return_does_not_reverse(self):
        replay = Replay().feed(0.5)
        replay.feed(0.3, dy=lambda t: -0.035 * t).feed(0.75, dy=-0.035)
        replay.feed(0.8, dy=lambda t: -0.035 * (1 - t)).feed(0.7)
        self.assertEqual(replay.types(), ['SWIPE_UP'])
        self.assertEqual(replay.tracker.state, 'READY')

    def test_scale_and_sensitivity(self):
        for scale in (0.5, 1, 1.7):
            for sensitivity in (0.6, 1, 2):
                replay = Replay(scale=scale, sensitivity=sensitivity).feed(0.5)
                replay.feed(0.4, dy=lambda t: -0.035 * t)
                replay.feed(0.3, dy=lambda t: -0.035 * (1 - t)).feed(0.5)
                self.assertEqual(replay.types(), ['SWIPE_UP'])

    def test_idle_jitter_and_single_frame_spikes(self):
        replay = Replay().feed(0.5)
        replay.feed(8, dy=lambda t: 0.0018 * math.sin(t * 330), dx=lambda t: 0.0015 * math.cos(t * 250))
        for finger in (8, 12, 20, 8):
            replay.feed(1 / 30, finger=finger, dy=-0.045).feed(0.6)
        self.assertEqual(replay.types(), [])

    def test_horizontal_motion_and_ambiguous_opposing_strokes(self):
        replay = Replay().feed(0.5)
        replay.feed(0.4, dy=lambda t: -0.012 * t, dx=lambda t: 0.09 * t)
        replay.feed(0.4, dy=lambda t: -0.012 * (1 - t), dx=lambda t: 0.09 * (1 - t)).feed(0.6)
        self.assertEqual(replay.types(), [])
        tracker = RelaxedIntentTracker()
        events = []
        for i in range(70):
            offset = 0 if i < 15 or i > 40 else min((i - 15) / 10, (40 - i) / 10) * 0.03
            event = tracker.update(make_hand(fingers={8: (0, -offset), 12: (0, offset)}), 1000 + i / 30)
            if event:
                events.append(event)
        self.assertEqual(events, [])

    def test_hand_entering_view_and_tracking_gap(self):
        replay = Replay()
        replay.feed(0.6, hand_y=lambda t: 0.12 * (1 - t)).feed(0.5)
        self.assertEqual(replay.types(), [])
        replay.feed(0.15, dy=lambda t: -0.02 * t)
        replay.now += 0.4
        replay.feed(0.5, dy=-0.02)
        self.assertEqual(replay.types(), [])
        self.assertEqual(replay.tracker.state, 'READY')

    def test_invalid_landmarks_recalibrate(self):
        replay = Replay().feed(0.5)
        bad = make_hand()
        bad[8].y = float('nan')
        self.assertIsNone(replay.tracker.update(bad, replay.now))
        self.assertEqual(replay.tracker.state, 'CALIBRATING')


@__import__('unittest.mock').mock.patch('app.gesture_engine.MOTION_GESTURES_ENABLED', True)
class PipelineTests(unittest.TestCase):
    def test_landmarks_to_gesture_and_loss_feedback(self):
        fake = SimpleNamespace(detect_for_video=lambda image, timestamp: SimpleNamespace(hand_landmarks=[make_hand()]), close=lambda: None)
        engine = GestureEngine(landmarker=fake)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        events = []
        for i in range(75):
            dy = 0 if i < 20 or i > 45 else -0.03 * min((i - 20) / 12, (45 - i) / 12)
            fake.detect_for_video = lambda image, timestamp, dy=dy: SimpleNamespace(hand_landmarks=[make_hand(fingers={12: (0, dy)})])
            result = engine.process_frame(frame, now=1000 + i / 30)
            if result:
                events.append(result.type)
        self.assertEqual(events, ['SWIPE_UP'])
        fake.detect_for_video = lambda image, timestamp: SimpleNamespace(hand_landmarks=[])
        self.assertIsNone(engine.process_frame(frame, now=1003))
        self.assertFalse(engine.get_hand_info()['tracking'])
        self.assertEqual(engine.hand_state.intent.state, 'CALIBRATING')
        engine.cleanup()

    def test_background_motion_without_hand_never_scrolls(self):
        fake = SimpleNamespace(detect_for_video=lambda image, timestamp: SimpleNamespace(hand_landmarks=[]), close=lambda: None)
        engine = GestureEngine(landmarker=fake)
        rng = np.random.default_rng(17)
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        for i in range(25):
            self.assertIsNone(engine.process_frame(np.roll(frame, i * 3, axis=0), now=1000 + i / 30))
        self.assertEqual(engine.gesture_count, 0)

    def test_changing_hands_requires_a_new_neutral_position(self):
        fake = SimpleNamespace(close=lambda: None)
        engine = GestureEngine(landmarker=fake)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for i in range(20):
            fake.detect_for_video = lambda image, timestamp: SimpleNamespace(
                hand_landmarks=[make_hand()],
                handedness=[[SimpleNamespace(category_name='Left', score=0.95)]])
            engine.process_frame(frame, now=1000 + i / 30)
        self.assertEqual(engine.hand_state.intent.state, 'READY')
        fake.detect_for_video = lambda image, timestamp: SimpleNamespace(
            hand_landmarks=[make_hand(hand_y=-0.09)],
            handedness=[[SimpleNamespace(category_name='Right', score=0.95)]])
        self.assertIsNone(engine.process_frame(frame, now=1000 + 20 / 30))
        self.assertEqual(engine.hand_state.intent.state, 'CALIBRATING')

    def test_camera_shake_and_hand_region_exclusion(self):
        rng = np.random.default_rng(9)
        texture = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
        guard = CameraMotionGuard()
        self.assertFalse(guard.update(texture, make_hand()))
        changed_hand = texture.copy()
        changed_hand[60:115, 45:110] = 90
        self.assertFalse(guard.update(changed_hand, make_hand()))
        guard.reset()
        guard.update(texture, make_hand())
        shifted = cv2.warpAffine(texture, np.float32([[1, 0, 3], [0, 1, -2]]), (160, 120))
        self.assertTrue(guard.update(shifted, make_hand()))


class InputTests(unittest.TestCase):
    def setUp(self):
        for target, value in (('app.input_simulator._ensure_cursor_over_target', None),
                              ('app.input_simulator._get_foreground_handle', 123),
                              ('app.window_manager.activate_target_browser', False)):
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_native_input_struct_size(self):
        self.assertEqual(ctypes.sizeof(INPUT), 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)

    def test_phone_and_natural_mapping_for_wheel_and_keyboard(self):
        for phone in (True, False):
            for mode in ('scroll', 'keyboard'):
                for gesture in ('SWIPE_UP', 'SWIPE_DOWN'):
                    with patch('app.input_simulator._get_foreground_window_title', return_value='Browser page'), patch('app.input_simulator._send_input', return_value=True) as send:
                        sim = InputSimulator(mode, phone)
                        result = sim.execute_gesture(GestureResult(gesture))
                        self.assertTrue(result['success'])
                        down = (gesture == 'SWIPE_UP') == phone
                        if mode == 'scroll':
                            inp = send.call_args.args[0]
                            self.assertEqual(inp.type, INPUT_MOUSE)
                            self.assertEqual(ctypes.c_int32(inp.union.mi.mouseData).value < 0, down)
                        else:
                            self.assertEqual(result['key'], 'down' if down else 'up')
                            self.assertEqual(send.call_count, 2)
                            self.assertEqual(send.call_args.args[0].type, INPUT_KEYBOARD)

    def test_input_failure_is_not_counted_as_success(self):
        with patch('app.input_simulator._get_foreground_window_title', return_value='Browser'), patch('app.input_simulator._send_input', return_value=False):
            sim = InputSimulator('scroll')
            result = sim.execute_gesture(GestureResult('SWIPE_UP'))
            self.assertFalse(result['success'])
            self.assertEqual(sim.total_actions, 0)
            self.assertIn('failed', result['description'])

    def test_hud_focus_does_not_consume_input(self):
        with patch('app.input_simulator._get_foreground_window_title', return_value='HandScroller (Privacy Mode)'), patch('app.input_simulator._send_input') as send:
            result = InputSimulator().execute_gesture(GestureResult('SWIPE_DOWN'))
            self.assertFalse(result['success'])
            send.assert_not_called()

    def test_project_named_handscroller_can_receive_input(self):
        with patch('app.input_simulator._get_foreground_window_title', return_value='HandScroller - Codex'), patch('app.input_simulator._send_input', return_value=True):
            result = InputSimulator('scroll').execute_gesture(GestureResult('SWIPE_UP'))
            self.assertTrue(result['success'])

    def test_regular_youtube_uses_wheel(self):
        for title, expected in (('Video - YouTube', 'scroll'), ('Shorts - YouTube', 'keyboard'), ('TikTok', 'keyboard')):
            with patch('app.input_simulator._get_foreground_window_title', return_value=title.lower()):
                self.assertEqual(InputSimulator('smart')._get_effective_mode(), expected)


if __name__ == '__main__':
    unittest.main()
