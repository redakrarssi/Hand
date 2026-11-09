"""
Unit tests for TouchGestureDetector, InputSimulator touch handling,
and GestureEngine touch integration.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.gesture_engine import GestureEngine, GestureResult
from app.input_simulator import InputSimulator
from app.touch_detector import TouchGestureDetector, THUMB_TIP, INDEX_TIP, MIDDLE_TIP


def make_landmarks(thumb_pos=None, index_pos=None, middle_pos=None, wrist_pos=(0.5, 0.8), knuckle_pos=(0.5, 0.6)):
    """
    Create a 21-landmark mock hand.
    wrist_pos=(0.5, 0.8) and knuckle_pos=(0.5, 0.6) yields a hand_scale of 0.20.
    """
    pts = [SimpleNamespace(x=0.5, y=0.7, z=0.0) for _ in range(21)]
    # Wrist (0) and Middle MCP (9)
    pts[0].x, pts[0].y = wrist_pos
    pts[9].x, pts[9].y = knuckle_pos
    # Thumb Tip (4)
    if thumb_pos is not None:
        pts[THUMB_TIP].x, pts[THUMB_TIP].y = thumb_pos
    else:
        pts[THUMB_TIP].x, pts[THUMB_TIP].y = (0.38, 0.65)
    # Index Tip (8)
    if index_pos is not None:
        pts[INDEX_TIP].x, pts[INDEX_TIP].y = index_pos
    else:
        pts[INDEX_TIP].x, pts[INDEX_TIP].y = (0.46, 0.55)
    # Middle Tip (12)
    if middle_pos is not None:
        pts[MIDDLE_TIP].x, pts[MIDDLE_TIP].y = middle_pos
    else:
        pts[MIDDLE_TIP].x, pts[MIDDLE_TIP].y = (0.50, 0.53)
    return pts


class TestTouchGestureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TouchGestureDetector(sensitivity=1.0)

    def touch(self, landmarks, now):
        """Feed the two frames needed to confirm contact; return the result."""
        for i in range(2):
            self.assertIsNone(self.detector.update(landmarks, now=now + i * 0.033))
        return self.detector.update(landmarks, now=now + 0.066)

    def test_open_hand_no_touch(self):
        """Open hand with fingers apart should not trigger any touch gesture."""
        landmarks = make_landmarks(
            thumb_pos=(0.35, 0.65),
            index_pos=(0.46, 0.52),
            middle_pos=(0.52, 0.50)
        )
        res = self.detector.update(landmarks, now=10.0)
        self.assertIsNone(res)
        self.assertFalse(self.detector.touching_index)
        self.assertFalse(self.detector.touching_middle)
        self.assertIsNone(self.detector.active_touch)

    def test_thumb_touches_index_triggers_scroll_down(self):
        """When thumb tip touches index tip, SWIPE_DOWN with source touch_thumb_index is emitted."""
        # Touch point at (0.40, 0.58) for both thumb (4) and index (8)
        landmarks = make_landmarks(
            thumb_pos=(0.40, 0.58),
            index_pos=(0.402, 0.582),
            middle_pos=(0.54, 0.52)
        )
        res = self.touch(landmarks, 10.0)
        self.assertIsNotNone(res)
        self.assertEqual(res.type, 'SWIPE_DOWN')
        self.assertEqual(res.source, 'touch_thumb_index')
        self.assertTrue(self.detector.touching_index)
        self.assertFalse(self.detector.touching_middle)
        self.assertEqual(self.detector.active_touch, 'index')

    def test_thumb_touches_middle_triggers_scroll_up(self):
        """When thumb tip touches middle tip, SWIPE_UP with source touch_thumb_middle is emitted."""
        # Touch point at (0.48, 0.56) for both thumb (4) and middle (12)
        landmarks = make_landmarks(
            thumb_pos=(0.48, 0.56),
            index_pos=(0.40, 0.50),
            middle_pos=(0.482, 0.562)
        )
        res = self.touch(landmarks, 10.0)
        self.assertIsNotNone(res)
        self.assertEqual(res.type, 'SWIPE_UP')
        self.assertEqual(res.source, 'touch_thumb_middle')
        self.assertTrue(self.detector.touching_middle)
        self.assertFalse(self.detector.touching_index)
        self.assertEqual(self.detector.active_touch, 'middle')

    def test_single_frame_spike_is_ignored(self):
        landmarks = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.402, 0.582), middle_pos=(0.54, 0.52))
        open_lm = make_landmarks(thumb_pos=(0.35, 0.65), index_pos=(0.46, 0.52), middle_pos=(0.52, 0.50))
        self.assertIsNone(self.detector.update(landmarks, now=10.0))
        self.assertIsNone(self.detector.update(open_lm, now=10.033))
        self.assertIsNone(self.detector.update(landmarks, now=10.066))
        self.assertIsNone(self.detector.update(open_lm, now=10.1))
        self.assertIsNone(self.detector.active_touch)

    def test_hold_does_not_burst_scroll(self):
        """A normal pinch pages once; repeat only after a long deliberate hold."""
        landmarks = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.402, 0.582), middle_pos=(0.54, 0.52))
        self.assertIsNotNone(self.touch(landmarks, 10.0))
        for t in (10.1, 10.35, 10.6, 11.0):
            self.assertIsNone(self.detector.update(landmarks, now=t))
        # Long hold: slow repeat
        res = self.detector.update(landmarks, now=11.3)
        self.assertIsNotNone(res)
        self.assertEqual(res.type, 'SWIPE_DOWN')
        self.assertIsNone(self.detector.update(landmarks, now=11.6))

    def test_release_and_rearm(self):
        """A pinch after a confirmed release pages again, but not within the lockout."""
        touch_landmarks = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.402, 0.582), middle_pos=(0.54, 0.52))
        open_landmarks = make_landmarks(thumb_pos=(0.35, 0.65), index_pos=(0.46, 0.52), middle_pos=(0.52, 0.50))

        res1 = self.touch(touch_landmarks, 10.0)
        self.assertEqual(res1.type, 'SWIPE_DOWN')

        # Release needs two frames apart
        self.assertIsNone(self.detector.update(open_landmarks, now=10.1))
        self.assertIsNone(self.detector.update(open_landmarks, now=10.133))
        self.assertIsNone(self.detector.active_touch)

        # Immediate re-pinch inside the lockout is swallowed (no double page)
        self.assertIsNone(self.touch(touch_landmarks, 10.2))
        self.assertIsNone(self.detector.update(touch_landmarks, now=10.3))

        # Release then a later pinch pages again
        self.detector.update(open_landmarks, now=10.4)
        self.detector.update(open_landmarks, now=10.433)
        res3 = self.touch(touch_landmarks, 10.8)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.type, 'SWIPE_DOWN')

    def test_small_gap_is_not_a_touch(self):
        """Fingers close but with a visible gap (norm ~0.26) must not scroll."""
        gap = make_landmarks(thumb_pos=(0.40, 0.60), index_pos=(0.452, 0.60), middle_pos=(0.55, 0.60))
        for i in range(30):
            self.assertIsNone(self.detector.update(gap, now=10.0 + i * 0.033))
        self.assertIsNone(self.detector.active_touch)

    def test_ambiguous_pinch_is_ignored(self):
        landmarks = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.44, 0.58), middle_pos=(0.44, 0.582))
        self.detector.update(landmarks, now=10.0)
        self.assertIsNone(self.detector.update(landmarks, now=10.033))
        self.assertIsNone(self.detector.active_touch)

    def test_hysteresis_band(self):
        """Distance between touch_threshold and release_threshold maintains current state."""
        # Enter threshold is ~0.28, exit threshold is ~0.38
        # hand_scale is 0.20
        # distance = 0.05 / 0.20 = 0.25 (below 0.28 -> enters contact)
        contact_lm = make_landmarks(
            thumb_pos=(0.40, 0.60),
            index_pos=(0.44, 0.60),  # dist = 0.04 -> norm = 0.20
            middle_pos=(0.55, 0.60)
        )
        self.assertIsNotNone(self.touch(contact_lm, 1.0))
        self.assertEqual(self.detector.active_touch, 'index')

        # Fingers slightly loosen: distance = 0.065 / 0.20 = 0.325
        # (between 0.28 and 0.38) -> should STAY in active_touch='index' due to hysteresis
        loosen_lm = make_landmarks(
            thumb_pos=(0.40, 0.60),
            index_pos=(0.455, 0.60),  # dist = 0.055 -> norm = 0.275
            middle_pos=(0.55, 0.60)
        )
        self.detector.update(loosen_lm, now=1.2)
        self.assertEqual(self.detector.active_touch, 'index')

        # Fingers separate: distance = 0.085 / 0.20 = 0.425 (> 0.38)
        separate_lm = make_landmarks(
            thumb_pos=(0.40, 0.60),
            index_pos=(0.47, 0.60),  # dist = 0.07 -> norm = 0.35
            middle_pos=(0.55, 0.60)
        )
        self.detector.update(separate_lm, now=1.3)
        self.detector.update(separate_lm, now=1.333)
        self.assertIsNone(self.detector.active_touch)

    def test_fingers_close_enough_hovering_does_not_engage(self):
        """Fingers hovering close (~0.30 norm) must not trigger touch or engage the lockout."""
        hover_lm = make_landmarks(
            thumb_pos=(0.40, 0.60),
            index_pos=(0.46, 0.60),  # dist = 0.06 / 0.20 = 0.30 norm
            middle_pos=(0.55, 0.60)
        )
        for i in range(10):
            res = self.detector.update(hover_lm, now=10.0 + i * 0.033)
            self.assertIsNone(res)
        self.assertIsNone(self.detector.active_touch)
        # Crucial fix: is_engaged must be False so swipe scrolling is not blocked!
        self.assertFalse(self.detector.is_engaged(now=10.5))

    def test_adjacent_middle_curl_does_not_block_index_touch(self):
        """When thumb touches index, a naturally curled middle finger must not veto the touch."""
        # Index is in touch range (0.03 / 0.20 = 0.15 norm), middle is slightly curled (0.045 / 0.20 = 0.225 norm)
        curled_lm = make_landmarks(
            thumb_pos=(0.40, 0.60),
            index_pos=(0.43, 0.60),
            middle_pos=(0.445, 0.60)
        )
        res = self.touch(curled_lm, 10.0)
        self.assertIsNotNone(res)
        self.assertEqual(res.type, 'SWIPE_DOWN')
        self.assertEqual(self.detector.active_touch, 'index')

    def test_cooldown_held_pinch_fires_when_interval_elapses(self):
        """A pinch starting during cooldown does not freeze; it fires as soon as interval elapses."""
        touch_lm = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.402, 0.582), middle_pos=(0.54, 0.52))
        open_lm = make_landmarks(thumb_pos=(0.35, 0.65), index_pos=(0.46, 0.52), middle_pos=(0.52, 0.50))

        # First touch fires at 10.0s
        self.assertIsNotNone(self.touch(touch_lm, 10.0))

        # Release hand
        self.detector.update(open_lm, now=10.05)
        self.detector.update(open_lm, now=10.08)
        self.assertIsNone(self.detector.active_touch)

        # Repinch quickly at 10.15s (within MIN_FIRE_INTERVAL of 0.30s)
        # 3 candidate frames transition into active_touch, but _fired is False due to cooldown
        self.assertIsNone(self.detector.update(touch_lm, now=10.15))
        self.assertIsNone(self.detector.update(touch_lm, now=10.18))
        self.assertIsNone(self.detector.update(touch_lm, now=10.21))
        self.assertEqual(self.detector.active_touch, 'index')
        self.assertFalse(self.detector._fired)

        # Crucial fix: as soon as now - last_fire_time >= 0.30s (i.e. at 10.40s), held pinch FIRES!
        res = self.detector.update(touch_lm, now=10.40)
        self.assertIsNotNone(res)
        self.assertEqual(res.type, 'SWIPE_DOWN')
        self.assertTrue(self.detector._fired)

    def test_candidate_single_frame_noise_tolerance(self):
        """A single dropped frame during onset should not completely wipe candidate progress."""
        touch_lm = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.402, 0.582), middle_pos=(0.54, 0.52))
        open_lm = make_landmarks(thumb_pos=(0.35, 0.65), index_pos=(0.46, 0.52), middle_pos=(0.52, 0.50))

        # Frame 1: Candidate detected (candidate_frames=1)
        self.assertIsNone(self.detector.update(touch_lm, now=1.0))
        self.assertEqual(self.detector._candidate, 'index')
        self.assertEqual(self.detector._candidate_frames, 1)

        # Frame 2: Momentary camera noise dropout (candidate=None)
        self.assertIsNone(self.detector.update(open_lm, now=1.033))
        # 1 grace frame preserves candidate progress
        self.assertEqual(self.detector._candidate_frames, 1)

        # Frame 3: Candidate detected again (candidate_frames=2)
        self.assertIsNone(self.detector.update(touch_lm, now=1.066))
        self.assertEqual(self.detector._candidate_frames, 2)

        # Frame 4: Candidate confirmed (candidate_frames=3 -> fires!)
        res = self.detector.update(touch_lm, now=1.100)
        self.assertIsNotNone(res)
        self.assertEqual(res.type, 'SWIPE_DOWN')



class TestInputSimulatorTouch(unittest.TestCase):
    def setUp(self):
        self.sim = InputSimulator(mode='scroll')
        # Stub Windows SendInput calls
        self.sim._simulate_key_press = MagicMock(return_value=True)
        self.sim._simulate_scroll = MagicMock(return_value=True)

    @patch('app.input_simulator._get_foreground_handle', return_value=1234)
    @patch('app.input_simulator._get_foreground_window_title', return_value='Google Chrome')
    @patch('app.window_manager.target_error', return_value=None)
    def test_thumb_index_scrolls_down_in_scroll_mode(self, mock_err, mock_title, mock_handle):
        """Thumb + Index touch executes scroll down (-1) in mouse wheel mode."""
        gesture = GestureResult('SWIPE_DOWN', source='touch_thumb_index')
        action = self.sim.execute_gesture(gesture)
        self.assertTrue(action['success'])
        self.assertEqual(action['scroll'], 'down')
        self.sim._simulate_scroll.assert_called_with(-1)

    @patch('app.input_simulator._get_foreground_handle', return_value=1234)
    @patch('app.input_simulator._get_foreground_window_title', return_value='Google Chrome')
    @patch('app.window_manager.target_error', return_value=None)
    def test_thumb_middle_scrolls_up_in_scroll_mode(self, mock_err, mock_title, mock_handle):
        """Thumb + Middle touch executes scroll up (1) in mouse wheel mode."""
        gesture = GestureResult('SWIPE_UP', source='touch_thumb_middle')
        action = self.sim.execute_gesture(gesture)
        self.assertTrue(action['success'])
        self.assertEqual(action['scroll'], 'up')
        self.sim._simulate_scroll.assert_called_with(1)

    @patch('app.input_simulator._get_foreground_handle', return_value=1234)
    @patch('app.input_simulator._get_foreground_window_title', return_value='TikTok - Watch')
    @patch('app.window_manager.target_error', return_value=None)
    def test_thumb_index_presses_down_key_in_keyboard_mode(self, mock_err, mock_title, mock_handle):
        """Thumb + Index touch presses 'down' key in keyboard mode."""
        self.sim.mode = 'keyboard'
        gesture = GestureResult('SWIPE_DOWN', source='touch_thumb_index')
        action = self.sim.execute_gesture(gesture)
        self.assertTrue(action['success'])
        self.assertEqual(action['key'], 'down')
        self.sim._simulate_key_press.assert_called_with('down')

    @patch('app.input_simulator._get_foreground_handle', return_value=1234)
    @patch('app.input_simulator._get_foreground_window_title', return_value='TikTok - Watch')
    @patch('app.window_manager.target_error', return_value=None)
    def test_thumb_middle_presses_up_key_in_keyboard_mode(self, mock_err, mock_title, mock_handle):
        """Thumb + Middle touch presses 'up' key in keyboard mode."""
        self.sim.mode = 'keyboard'
        gesture = GestureResult('SWIPE_UP', source='touch_thumb_middle')
        action = self.sim.execute_gesture(gesture)
        self.assertTrue(action['success'])
        self.assertEqual(action['key'], 'up')
        self.sim._simulate_key_press.assert_called_with('up')

    @patch('app.input_simulator._get_foreground_handle', return_value=1234)
    @patch('app.input_simulator._get_foreground_window_title', return_value='Google Chrome')
    @patch('app.window_manager.target_error', return_value=None)
    def test_touch_independent_of_phone_style(self, mock_err, mock_title, mock_handle):
        """Touch gesture direction is direct and never inverted by phone_style."""
        self.sim.phone_style = True  # Phone style would invert swipe, but not touch
        gesture_down = GestureResult('SWIPE_DOWN', source='touch_thumb_index')
        action_down = self.sim.execute_gesture(gesture_down)
        self.assertEqual(action_down['scroll'], 'down')
        self.sim._simulate_scroll.assert_called_with(-1)

        self.sim._simulate_scroll.reset_mock()
        gesture_up = GestureResult('SWIPE_UP', source='touch_thumb_middle')
        action_up = self.sim.execute_gesture(gesture_up)
        self.assertEqual(action_up['scroll'], 'up')
        self.sim._simulate_scroll.assert_called_with(1)


class TestGestureEngineTouchIntegration(unittest.TestCase):
    def test_engine_processes_touch_and_updates_status(self):
        """GestureEngine should detect touch and reflect it in get_hand_info()."""
        # Create engine with mock landmarker
        mock_landmarker = MagicMock()
        engine = GestureEngine(landmarker=mock_landmarker)

        # Mock frame
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Touch landmarks: Thumb + Index
        touch_lm = make_landmarks(
            thumb_pos=(0.40, 0.58),
            index_pos=(0.402, 0.582),
            middle_pos=(0.54, 0.52)
        )
        mock_result = SimpleNamespace(
            hand_landmarks=[touch_lm],
            handedness=[[SimpleNamespace(score=0.95, category_name='Right')]]
        )
        mock_landmarker.detect_for_video.return_value = mock_result

        self.assertIsNone(engine.process_frame(frame, now=1.0))
        self.assertIsNone(engine.process_frame(frame, now=1.033))
        gesture = engine.process_frame(frame, now=1.066)
        self.assertIsNotNone(gesture)
        self.assertEqual(gesture.type, 'SWIPE_DOWN')
        self.assertEqual(gesture.source, 'touch_thumb_index')

        info = engine.get_hand_info()
        self.assertTrue(info['touching_index'])
        self.assertIn('DOWN', info['intent_status'])

    def test_touch_release_preserves_intent_tracker_baseline(self):
        """Touching and releasing fingers must preserve the intent tracker baseline without AttributeError."""
        import numpy as np
        mock_landmarker = MagicMock()
        engine = GestureEngine(landmarker=mock_landmarker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Calibrate intent tracker with open hand
        open_lm = make_landmarks(thumb_pos=(0.35, 0.65), index_pos=(0.46, 0.52), middle_pos=(0.52, 0.50))
        mock_landmarker.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[open_lm],
            handedness=[[SimpleNamespace(score=0.95, category_name='Right')]]
        )
        for t in np.linspace(1.0, 1.25, 10):
            engine.process_frame(frame, now=float(t))

        # 2. Touch fingers
        touch_lm = make_landmarks(thumb_pos=(0.40, 0.58), index_pos=(0.402, 0.582), middle_pos=(0.54, 0.52))
        mock_landmarker.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[touch_lm],
            handedness=[[SimpleNamespace(score=0.95, category_name='Right')]]
        )
        self.assertIsNone(engine.process_frame(frame, now=1.30))
        self.assertIsNone(engine.process_frame(frame, now=1.333))
        g = engine.process_frame(frame, now=1.366)
        self.assertIsNotNone(g)
        self.assertEqual(g.type, 'SWIPE_DOWN')

        # Baseline must still be a Sample object, not corrupted into a list
        from app.intent_tracker import Sample
        self.assertIsInstance(engine.hand_state.intent.baseline, Sample)

        # 3. Release fingers (must not crash with AttributeError)
        mock_landmarker.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[open_lm],
            handedness=[[SimpleNamespace(score=0.95, category_name='Right')]]
        )
        g_release = engine.process_frame(frame, now=1.45)
        # Should execute cleanly without error
        self.assertIsNone(g_release)


class TestHandMovementIgnored(unittest.TestCase):
    def test_moving_hand_and_fingers_never_scrolls(self):
        import numpy as np
        mock_landmarker = MagicMock()
        engine = GestureEngine(landmarker=mock_landmarker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gestures = []
        for i in range(120):
            dy = 0.15 * math.sin(i / 6)
            lm = make_landmarks(thumb_pos=(0.35, 0.65 + dy), index_pos=(0.46, 0.52 + dy),
                                middle_pos=(0.52, 0.50 + dy), wrist_pos=(0.5, 0.8 + dy),
                                knuckle_pos=(0.5, 0.6 + dy))
            mock_landmarker.detect_for_video.return_value = SimpleNamespace(
                hand_landmarks=[lm], handedness=[[SimpleNamespace(score=0.95, category_name='Right')]])
            gestures.append(engine.process_frame(frame, now=1.0 + i * 0.033))
        self.assertEqual([g for g in gestures if g], [])


if __name__ == '__main__':
    unittest.main()
