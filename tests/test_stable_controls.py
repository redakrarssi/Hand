"""User controls stay predictable through gestures, feedback, and restarts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.gesture_engine import GestureEngine, GestureResult
from app.input_simulator import InputSimulator
from app.learning_curve import UserProfile, load_profile
from app.adaptive_learner import AdaptiveLearner
from main import apply_control_preferences, parse_args, remember_controls
from test_relaxed_scrolling import make_hand


@__import__('unittest.mock').mock.patch('app.gesture_engine.MOTION_GESTURES_ENABLED', True)
class StableControlsTests(unittest.TestCase):
    def engine(self, gesture_undo=False):
        fake = SimpleNamespace(detect_for_video=lambda image, timestamp: SimpleNamespace(hand_landmarks=[make_hand()]), close=lambda: None)
        engine = GestureEngine(fake, gesture_undo=gesture_undo)
        self.addCleanup(engine.cleanup)
        return engine

    def test_explicit_direction_and_sensitivity_override_saved_preferences(self):
        engine = self.engine()
        sim = InputSimulator('scroll')
        profile = UserProfile(calibrated=True, phone_style=False, sensitivity=0.8)
        args = SimpleNamespace(direction='phone', sensitivity=1.7)
        apply_control_preferences(engine, sim, profile, args)
        self.assertTrue(sim.phone_style)
        self.assertEqual(engine.sensitivity, 1.7)

    def test_saved_preferences_are_kept_when_no_override_is_supplied(self):
        engine = self.engine()
        sim = InputSimulator('scroll')
        profile = UserProfile(calibrated=False, phone_style=False, sensitivity=1.4)
        apply_control_preferences(engine, sim, profile, SimpleNamespace(direction=None, sensitivity=None))
        self.assertFalse(sim.phone_style)
        self.assertEqual(engine.sensitivity, 1.4)

    def test_explicit_control_changes_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.learning_curve.PROFILE_PATH', str(Path(directory) / 'profile.json')):
            engine = self.engine()
            sim = InputSimulator('scroll', phone_style=False)
            engine.set_sensitivity(1.6)
            self.assertTrue(remember_controls(engine, sim))
            next_engine = self.engine()
            next_sim = InputSimulator('scroll')
            apply_control_preferences(next_engine, next_sim, load_profile(), SimpleNamespace(direction=None, sensitivity=None))
            self.assertFalse(next_sim.phone_style)
            self.assertEqual(next_engine.sensitivity, 1.6)

    def test_sideways_wag_has_no_special_action_by_default(self):
        import math
        for enabled in (False, True):
            engine = self.engine(gesture_undo=enabled)
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            events = []
            for i in range(30):
                hand = make_hand(fingers={8: (0.045 * math.sin(i * 0.75), 0)})
                engine.landmarker.detect_for_video = lambda image, timestamp, hand=hand: SimpleNamespace(hand_landmarks=[hand])
                event = engine.process_frame(frame, now=1000 + i / 30)
                if event:
                    events.append(event.type)
            self.assertEqual('NO_NO' in events, enabled)
            self.assertNotIn('SWIPE_UP', events)
            self.assertNotIn('SWIPE_DOWN', events)

    def test_full_smooth_scroll_feedback_undoes_exact_distance(self):
        deltas = []
        with patch('app.input_simulator._get_foreground_handle', return_value=123), patch('app.input_simulator._get_foreground_window_title', return_value='Browser'), patch('app.input_simulator._ensure_cursor_over_target'), patch.object(InputSimulator, '_send_wheel_delta', side_effect=lambda value: deltas.append(value) or True):
            sim = InputSimulator('scroll')
            feedback = AdaptiveLearner()
            gesture = GestureResult('SWIPE_UP')
            action = sim.execute_gesture(gesture)
            feedback.record_action(action, gesture)
            sim.update_scroll(sim._wheel.started + 1)
            self.assertEqual(sum(deltas), sim.scroll_amount * 120)
            self.assertTrue(feedback.handle_correction(self.engine(), sim)['undone'])
            self.assertEqual(sum(deltas), 0)

    def test_undo_cannot_scroll_a_different_focused_app(self):
        with patch('app.input_simulator._get_foreground_handle', return_value=456), patch.object(InputSimulator, '_send_wheel_delta') as send:
            sim = InputSimulator('scroll')
            self.assertFalse(sim.undo_action({'success': True, 'target_window': 123, 'wheel_units': 960}))
            send.assert_not_called()

    def test_gesture_undo_requires_opt_in_and_direction_has_no_implicit_override(self):
        with patch('sys.argv', ['main.py']):
            args = parse_args()
            self.assertFalse(args.gesture_undo)
            self.assertIsNone(args.direction)
            self.assertIsNone(args.sensitivity)
        with patch('sys.argv', ['main.py', '--gesture-undo', '--direction', 'natural', '--sensitivity', '1.4']):
            args = parse_args()
            self.assertTrue(args.gesture_undo)
            self.assertEqual(args.direction, 'natural')
            self.assertEqual(args.sensitivity, 1.4)


if __name__ == '__main__':
    unittest.main()
