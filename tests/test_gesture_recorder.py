"""Unit tests for GestureRecorder and Reference-Based Gesture Recognition."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from app.dense_hand_tracker import DenseHandTracker
from app.gesture_engine import GestureEngine, GestureResult
from app.gesture_recorder import (
    GestureRecorder, GestureReferences, HandPoseSnapshot,
    load_references, save_references, REFERENCES_PATH, PROFILE_PATH
)
from tests.test_relaxed_scrolling import make_hand


class GestureRecorderTests(unittest.TestCase):
    """Verify 6-position reference recording and persistence."""

    def test_recorder_step_progression_via_manual_capture(self):
        """Test recording through all 7 steps (Rest + 3 UP + 3 DOWN) using manual capture."""
        completed_refs = []
        recorder = GestureRecorder(on_complete_callback=lambda refs: completed_refs.append(refs))
        recorder.start()
        self.assertTrue(recorder.is_active())

        # Step 0: REST
        hud_info = recorder.update(make_hand(hand_y=0.0), now=1000.0)
        self.assertEqual(hud_info['step_id'], 'REST')
        self.assertTrue(recorder.manual_capture())
        self.assertTrue(recorder.in_transition)

        # Transition to Step 1: UP_1
        recorder.update(make_hand(hand_y=0.0), now=1001.0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_1')

        # Record UP_1 (slight up: hand_y=-0.02)
        recorder.update(make_hand(hand_y=-0.02), now=1002.0)
        self.assertTrue(recorder.manual_capture())

        # Transition to Step 2: UP_2
        recorder.update(make_hand(hand_y=-0.02), now=1003.0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_2')

        # Record UP_2 (medium up: hand_y=-0.04)
        recorder.update(make_hand(hand_y=-0.04), now=1004.0)
        self.assertTrue(recorder.manual_capture())

        # Transition to Step 3: UP_3
        recorder.update(make_hand(hand_y=-0.04), now=1005.0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_3')

        # Record UP_3 (full up: hand_y=-0.07)
        recorder.update(make_hand(hand_y=-0.07), now=1006.0)
        self.assertTrue(recorder.manual_capture())

        # Transition to Step 4: DOWN_1
        recorder.update(make_hand(hand_y=-0.07), now=1007.0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'DOWN_1')

        # Record DOWN_1 (slight down: hand_y=0.02)
        recorder.update(make_hand(hand_y=0.02), now=1008.0)
        self.assertTrue(recorder.manual_capture())

        # Transition to Step 5: DOWN_2
        recorder.update(make_hand(hand_y=0.02), now=1009.0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'DOWN_2')

        # Record DOWN_2 (medium down: hand_y=0.04)
        recorder.update(make_hand(hand_y=0.04), now=1010.0)
        self.assertTrue(recorder.manual_capture())

        # Transition to Step 6: DOWN_3
        recorder.update(make_hand(hand_y=0.04), now=1011.0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'DOWN_3')

        # Record DOWN_3 (full down: hand_y=0.07)
        recorder.update(make_hand(hand_y=0.07), now=1012.0)
        self.assertTrue(recorder.manual_capture())

        # Final transition to complete
        with patch('app.gesture_recorder.save_references'):
            recorder.update(make_hand(hand_y=0.07), now=1013.0)

        self.assertFalse(recorder.is_active())
        self.assertEqual(len(completed_refs), 1)
        refs = completed_refs[0]
        self.assertTrue(refs.is_valid)
        self.assertEqual(len(refs.up_positions), 3)
        self.assertEqual(len(refs.down_positions), 3)
        self.assertIsNotNone(refs.rest_baseline)
        self.assertGreater(refs.up_threshold, 0)
        self.assertGreater(refs.down_threshold, 0)

    def test_recorder_hold_key_rest_calibration(self):
        """Test that holding key for hold duration calibrates resting neutral baseline."""
        recorder = GestureRecorder()
        recorder.start()

        # Hold key for > 0.85s while resting hand
        for i in range(35):
            hud = recorder.update(make_hand(hand_y=0.0), now=1000.0 + i * 0.033, key_held=True)

        # After holding key, Step REST should be locked into transition
        self.assertTrue(recorder.in_transition)
        self.assertIsNotNone(recorder.rest_baseline)

    def test_motion_without_key_does_not_start_recording(self):
        """Test that hand motion without holding key does NOT start recording or finalize."""
        recorder = GestureRecorder()
        recorder.start()

        # Step 0: establish rest baseline
        recorder.update(make_hand(hand_y=0.0), now=1000.0)
        recorder.manual_capture(now=1000.0)
        recorder.update(make_hand(hand_y=0.0), now=1001.0, key_held=False) # transition to UP_1
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_1')

        # Hand moves up Decisively without holding key
        recorder.update(make_hand(hand_y=-0.050), now=1001.033, key_held=False)
        recorder.update(make_hand(hand_y=-0.080), now=1001.066, key_held=False)
        recorder.update(make_hand(hand_y=-0.020), now=1002.500, key_held=False)

        # Recording should NOT have started, and no stroke saved
        self.assertFalse(recorder.is_recording_stroke)
        self.assertFalse(recorder.in_transition)
        self.assertEqual(len(recorder.recorded_up_trajectories), 0)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_1')

    def test_hold_key_push_to_record_stroke(self):
        """Test push-to-record where holding key streams movement and releasing completes it."""
        recorder = GestureRecorder()
        recorder.start()

        # Step 0: Calibrate rest baseline by holding key for 0.9s
        recorder.update(make_hand(hand_y=0.0), now=1000.0, key_held=True)
        recorder.update(make_hand(hand_y=0.0), now=1000.5, key_held=True)
        recorder.update(make_hand(hand_y=0.0), now=1001.0, key_held=True)

        self.assertTrue(recorder.in_transition)
        self.assertIsNotNone(recorder.rest_baseline)

        # Transition to Step 1: UP_1
        recorder.update(make_hand(hand_y=0.0), now=1001.7, key_held=False)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_1')

        # User presses down SPACE to start recording UP_1
        recorder.update(make_hand(hand_y=-0.010), now=1002.0, key_held=True)
        self.assertTrue(recorder.is_recording_stroke)

        # User reaches peak flick while holding SPACE
        recorder.update(make_hand(hand_y=-0.045), now=1002.15, key_held=True)
        self.assertTrue(recorder.is_recording_stroke)
        self.assertAlmostEqual(recorder.stroke_peak_disp, 0.045, places=3)

        # User finishes movement and releases SPACE
        recorder.update(make_hand(hand_y=-0.025), now=1002.30, key_held=False)

        # Stroke should be immediately finalized upon key release!
        self.assertFalse(recorder.is_recording_stroke)
        self.assertTrue(recorder.in_transition)
        self.assertEqual(len(recorder.recorded_up_trajectories), 1)
        traj = recorder.recorded_up_trajectories[0]
        self.assertEqual(traj['direction'], 'UP')
        self.assertAlmostEqual(traj['peak_displacement'], 0.045, places=3)
        self.assertGreater(traj['duration'], 0.10)

    def test_push_to_record_full_workflow_requires_explicit_press_and_release_per_step(self):
        """Verify each step requires an explicit press to start and release to finish."""
        recorder = GestureRecorder()
        recorder.start()

        # Step 0: REST
        recorder.update(make_hand(hand_y=0.0), now=100.0, key_held=True)
        recorder.update(make_hand(hand_y=0.0), now=101.0, key_held=True)
        self.assertTrue(recorder.in_transition)

        # Release key during transition
        recorder.update(make_hand(hand_y=0.0), now=101.8, key_held=False)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_1')

        # While key is not held, moving hand does NOT record
        recorder.update(make_hand(hand_y=-0.040), now=102.0, key_held=False)
        self.assertFalse(recorder.is_recording_stroke)

        # Press key down -> starts recording UP_1
        recorder.update(make_hand(hand_y=-0.010), now=103.0, key_held=True)
        self.assertTrue(recorder.is_recording_stroke)

        # Stream peak move while holding key
        recorder.update(make_hand(hand_y=-0.035), now=103.2, key_held=True)
        self.assertTrue(recorder.is_recording_stroke)

        # Release key -> finishes UP_1
        recorder.update(make_hand(hand_y=-0.020), now=103.4, key_held=False)
        self.assertFalse(recorder.is_recording_stroke)
        self.assertTrue(recorder.in_transition)
        self.assertEqual(len(recorder.recorded_up_trajectories), 1)

        # Advance past transition
        recorder.update(make_hand(hand_y=0.0), now=104.2, key_held=False)
        self.assertEqual(recorder.STEPS[recorder.current_step_idx]['id'], 'UP_2')

        # Holding key from before without release prevents premature start
        recorder.waiting_for_release = True
        hud = recorder.update(make_hand(hand_y=0.0), now=104.3, key_held=True)
        self.assertTrue(hud['waiting_for_release'])
        self.assertFalse(recorder.is_recording_stroke)

        # Release key -> clears waiting_for_release
        hud = recorder.update(make_hand(hand_y=0.0), now=104.4, key_held=False)
        self.assertFalse(hud['waiting_for_release'])

    def test_save_and_load_references_persistence(self):
        """Verify references are serialized and deserialized faithfully to/from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "gesture_references.json")
            prof_path = os.path.join(tmpdir, "user_profile.json")

            with patch('app.gesture_recorder.REFERENCES_PATH', ref_path), \
                 patch('app.gesture_recorder.PROFILE_PATH', prof_path):

                refs = GestureReferences(
                    version=1,
                    recorded_at=1700000000.0,
                    is_valid=True,
                    rest_baseline={'hand_y': 0.7, 'finger_y': 0.58, 'scale': 0.16},
                    up_positions=[
                        {'position_num': 1, 'disp_index': -0.02},
                        {'position_num': 2, 'disp_index': -0.04},
                        {'position_num': 3, 'disp_index': -0.06},
                    ],
                    down_positions=[
                        {'position_num': 1, 'disp_index': 0.02},
                        {'position_num': 2, 'disp_index': 0.04},
                        {'position_num': 3, 'disp_index': 0.06},
                    ],
                    up_threshold=0.018,
                    down_threshold=0.019,
                    primary_channel='index',
                    up_displacements=[0.02, 0.04, 0.06],
                    down_displacements=[0.02, 0.04, 0.06],
                )

                self.assertTrue(save_references(refs))
                self.assertTrue(os.path.exists(ref_path))
                self.assertTrue(os.path.exists(prof_path))

                loaded = load_references()
                self.assertTrue(loaded.is_valid)
                self.assertEqual(len(loaded.up_positions), 3)
                self.assertEqual(len(loaded.down_positions), 3)
                self.assertAlmostEqual(loaded.up_threshold, 0.018)
                self.assertAlmostEqual(loaded.down_threshold, 0.019)
                self.assertEqual(loaded.primary_channel, 'index')


class DenseTrackerReferenceIntegrationTests(unittest.TestCase):
    """Verify DenseHandTracker utilizes personal reference positions as ground truth."""

    def test_dense_tracker_loads_and_applies_references(self):
        """DenseHandTracker adjusts thresholds and channel weights based on loaded references."""
        tracker = DenseHandTracker()

        refs = GestureReferences(
            is_valid=True,
            up_threshold=0.014,
            down_threshold=0.016,
            primary_channel='index',
            up_positions=[
                {'step_id': 'UP_1', 'landmarks_norm': [{'x': 0, 'y': 0}] * 21},
                {'step_id': 'UP_2', 'landmarks_norm': [{'x': 0, 'y': 0}] * 21},
                {'step_id': 'UP_3', 'landmarks_norm': [{'x': 0, 'y': 0}] * 21},
            ],
            down_positions=[
                {'step_id': 'DOWN_1', 'landmarks_norm': [{'x': 0, 'y': 0}] * 21},
                {'step_id': 'DOWN_2', 'landmarks_norm': [{'x': 0, 'y': 0}] * 21},
                {'step_id': 'DOWN_3', 'landmarks_norm': [{'x': 0, 'y': 0}] * 21},
            ],
        )

        tracker.load_references(refs)
        self.assertTrue(tracker.has_references)
        self.assertEqual(tracker.ref_up_threshold, 0.014)
        self.assertEqual(tracker.ref_down_threshold, 0.016)

        summary = tracker.get_summary()
        self.assertTrue(summary['has_references'])
        self.assertEqual(summary['ref_up_thresh'], 0.014)

    def test_gesture_engine_load_references_integration(self):
        """GestureEngine loads references and synchronizes with DenseHandTracker."""
        fake = SimpleNamespace(detect_for_video=lambda img, ts: SimpleNamespace(hand_landmarks=[make_hand()]), close=lambda: None)
        engine = GestureEngine(fake)

        refs = GestureReferences(
            is_valid=True,
            up_threshold=0.022,
            down_threshold=0.022,
            primary_channel='index',
            up_positions=[{'step_id': 'UP_1'} for _ in range(3)],
            down_positions=[{'step_id': 'DOWN_1'} for _ in range(3)],
        )

        engine.load_references(refs)
        self.assertTrue(engine.dense_tracker.has_references)
        self.assertEqual(engine.dense_tracker.ref_up_threshold, 0.022)
        engine.cleanup()


if __name__ == '__main__':
    unittest.main()
