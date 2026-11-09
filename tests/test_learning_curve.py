import tempfile
from pathlib import Path
from unittest.mock import patch
import os
import unittest
from types import SimpleNamespace
from app.learning_curve import LearningCurve, UserProfile, save_profile, load_profile, PROFILE_PATH, TEST_PAGE_PATH


def make_landmarks(hand_y=0.7, finger_y=0.5, scale=0.1):
    landmarks = []
    for i in range(21):
        landmarks.append(SimpleNamespace(x=0.5, y=hand_y, z=0.0))
    # Wrist
    landmarks[0] = SimpleNamespace(x=0.5, y=hand_y, z=0.0)
    # Middle knuckle
    landmarks[9] = SimpleNamespace(x=0.5, y=hand_y - scale, z=0.0)
    # Index tip
    landmarks[8] = SimpleNamespace(x=0.5, y=finger_y, z=0.0)
    return landmarks


class TestLearningCurve(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        profile_patch = patch('app.learning_curve.PROFILE_PATH', str(Path(directory.name) / 'profile.json'))
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        browser_patch = patch('app.learning_curve.open_test_page', return_value=True)
        browser_patch.start()
        self.addCleanup(browser_patch.stop)
        self.lc = LearningCurve()

    def test_test_page_exists(self):
        self.assertTrue(os.path.exists(TEST_PAGE_PATH), "scroll_test.html should exist")

    def test_save_and_load_profile(self):
        prof = UserProfile(calibrated=True, finger_distance=0.035, hand_distance=0.050)
        save_profile(prof)
        loaded = load_profile()
        self.assertTrue(loaded.calibrated)
        self.assertAlmostEqual(loaded.finger_distance, 0.035)

    def test_default_phase_duration_is_ten_seconds(self):
        lc = LearningCurve()
        self.assertEqual(lc.phase_duration, 10.0)

    def test_learning_curve_step_progression(self):
        completed = []
        lc = LearningCurve(on_complete_callback=lambda p: completed.append(p), phase_duration=1.5)
        lc.start()
        self.assertEqual(lc.state, LearningCurve.STATE_REST)

        # Feed resting frames across the 1.5s phase duration
        t = 100.0
        for _ in range(20):
            lc.update(make_landmarks(hand_y=0.7, finger_y=0.5, scale=0.1), t)
            t += 0.1
        self.assertEqual(lc.state, LearningCurve.STATE_DOWN)

        # Feed scroll down stroke across 1.5s phase duration (in phone style, hand moves up / lower Y)
        for i in range(16):
            lc.update(make_landmarks(hand_y=0.62, finger_y=0.42, scale=0.1), t)
            t += 0.1
        self.assertEqual(lc.state, LearningCurve.STATE_UP)

        # Feed scroll up stroke across 1.5s phase duration (in phone style, hand moves down / higher Y)
        for i in range(16):
            lc.update(make_landmarks(hand_y=0.78, finger_y=0.58, scale=0.1), t)
            t += 0.1
        self.assertEqual(lc.state, LearningCurve.STATE_COMPLETE)

        # Step complete after timeout triggers callback
        t += 2.0
        lc.update(make_landmarks(), t)
        self.assertEqual(lc.state, LearningCurve.STATE_INACTIVE)
        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0].calibrated)


if __name__ == '__main__':
    unittest.main()
