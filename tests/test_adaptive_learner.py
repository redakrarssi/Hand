import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import math
import time
from types import SimpleNamespace
from app.adaptive_learner import NoNoDetector, AdaptiveLearner
from app.learning_curve import UserProfile, save_profile, load_profile


def make_landmarks(x=0.5, y=0.5, wag_x=0.0):
    lms = [SimpleNamespace(x=x, y=y) for _ in range(21)]
    lms[0] = SimpleNamespace(x=x, y=y)          # wrist
    lms[9] = SimpleNamespace(x=x, y=y - 0.1)    # knuckle (scale = 0.1)
    lms[5] = SimpleNamespace(x=x - 0.02, y=y - 0.1)
    lms[8] = SimpleNamespace(x=x + wag_x, y=y - 0.18)  # index tip
    return lms


class DummyGestureEngine:
    def __init__(self):
        self.profile = None

    def apply_profile(self, profile):
        self.profile = profile


class DummyInputSimulator:
    def __init__(self):
        self.phone_style = True
        self.undone_actions = []
        self.simulated_scrolls = []
        self.simulated_keys = []

    def undo_action(self, action_info):
        self.undone_actions.append(action_info)
        return True

    def _simulate_scroll(self, direction):
        self.simulated_scrolls.append(direction)
        return True

    def _simulate_key_press(self, key):
        self.simulated_keys.append(key)
        return True


class TestAdaptiveLearner(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        profile_patch = patch('app.learning_curve.PROFILE_PATH', str(Path(directory.name) / 'profile.json'))
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        browser_patch = patch('app.learning_curve.open_test_page', return_value=True)
        browser_patch.start()
        self.addCleanup(browser_patch.stop)
        self.detector = NoNoDetector(cooldown=0.5)
        self.learner = AdaptiveLearner()
        # Save baseline profile
        save_profile(UserProfile(calibrated=True, finger_distance=0.035, hand_distance=0.050, phone_style=True))

    def test_no_no_detector_recognizes_horizontal_wag(self):
        t = 100.0
        detected = False
        # Wag side-to-side across 24 frames (~0.8s)
        for i in range(24):
            wag = 0.045 * math.sin(i * 0.75)
            lms = make_landmarks(wag_x=wag)
            if self.detector.update(lms, t):
                detected = True
                break
            t += 1/30
        self.assertTrue(detected, "NoNoDetector should recognize genuine horizontal wagging")

    def test_no_no_detector_rejects_vertical_flick(self):
        t = 100.0
        detected = False
        for i in range(25):
            y_disp = -0.06 * (i / 25.0)
            lms = make_landmarks(y=0.5 + y_disp, wag_x=0.0)
            if self.detector.update(lms, t):
                detected = True
                break
            t += 1/30
        self.assertFalse(detected, "NoNoDetector should not trigger on vertical flick")

    def test_no_no_detector_rejects_resting_jitter(self):
        t = 100.0
        detected = False
        for i in range(25):
            lms = make_landmarks(wag_x=0.001 * (i % 2))
            if self.detector.update(lms, t):
                detected = True
                break
            t += 1/30
        self.assertFalse(detected, "NoNoDetector should not trigger on resting jitter")

    def test_feedback_undoes_without_retuning_at_any_stroke_speed(self):
        for velocity in (0.20, 0.65):
            engine = DummyGestureEngine()
            input_sim = DummyInputSimulator()
            profile_before = load_profile()
            action = {'success': True, 'scroll': 'down', 'mode': 'scroll'}
            gesture = SimpleNamespace(type='SWIPE_UP', source='index', velocity=velocity, confidence=0.95)
            self.learner.record_action(action, gesture)
            res = self.learner.handle_correction(engine, input_sim)
            self.assertTrue(res['undone'])
            self.assertEqual(res['type'], 'undo')
            self.assertEqual(len(input_sim.undone_actions), 1)
            self.assertTrue(input_sim.phone_style)
            self.assertEqual(load_profile(), profile_before)
            self.assertIsNone(engine.profile)
            self.assertEqual(input_sim.simulated_scrolls, [])
            self.assertEqual(input_sim.simulated_keys, [])

    def test_repeated_wag_does_not_repeat_undo_or_flip_direction(self):
        engine = DummyGestureEngine()
        input_sim = DummyInputSimulator()
        profile_before = load_profile()
        action = {'success': True, 'scroll': 'up', 'mode': 'scroll'}
        self.learner.record_action(action, SimpleNamespace(type='SWIPE_DOWN'))
        self.assertTrue(self.learner.handle_correction(engine, input_sim)['undone'])
        self.assertFalse(self.learner.handle_correction(engine, input_sim)['undone'])
        self.assertEqual(len(input_sim.undone_actions), 1)
        self.assertEqual(load_profile(), profile_before)
        self.assertTrue(input_sim.phone_style)

    def test_live_scroll_amount_is_preserved_for_undo(self):
        input_sim = DummyInputSimulator()
        action = {'success': True, 'scroll': 'down', 'mode': 'scroll', 'wheel_units': -120}
        self.learner.record_action(action, SimpleNamespace(type='SWIPE_UP'))
        action['wheel_units'] = -960
        self.learner.handle_correction(DummyGestureEngine(), input_sim)
        self.assertEqual(input_sim.undone_actions[0]['wheel_units'], -960)

    def test_old_action_is_not_undone(self):
        input_sim = DummyInputSimulator()
        self.learner.record_action({'success': True}, SimpleNamespace(type='SWIPE_UP'))
        self.learner.last_action['time'] -= 3
        self.assertFalse(self.learner.handle_correction(DummyGestureEngine(), input_sim)['undone'])
        self.assertEqual(input_sim.undone_actions, [])


if __name__ == '__main__':
    unittest.main()
