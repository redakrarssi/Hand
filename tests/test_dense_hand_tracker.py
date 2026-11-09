"""
Unit tests for DenseHandTracker - 134-Point Dense Hand Mesh & Resting/Micro-Movement Detection.
"""

import math
import random
import unittest
import cv2
import numpy as np
from types import SimpleNamespace

from app.dense_hand_tracker import DenseHandTracker
from tests.test_relaxed_scrolling import make_hand


class DenseHandTrackerTests(unittest.TestCase):
    """Verifies dense tracking specifications and noisy camera resilience."""

    def setUp(self):
        self.tracker = DenseHandTracker(sensitivity=1.0)
        self.hand = make_hand()

    def test_point_count_exceeds_100_points(self):
        """Verifies at least 100 points, concentrated on dominant fingers."""
        points = self.tracker.generate_dense_mesh(self.hand)
        
        # 1. Total points must exceed 100
        self.assertGreaterEqual(len(points), 100)
        self.assertEqual(len(points), 134)

        # 2. Finger points must exceed 100 points
        finger_points = [p for p in points if p.region != 'palm']
        self.assertGreaterEqual(len(finger_points), 100)
        self.assertEqual(len(finger_points), 116)

        # 3. Dominant fingers (Index, Middle, Thumb) concentration
        index_pts = [p for p in points if p.region == 'index']
        middle_pts = [p for p in points if p.region == 'middle']
        thumb_pts = [p for p in points if p.region == 'thumb']
        dominant_pts = [p for p in points if p.is_dominant]

        self.assertEqual(len(index_pts), 32)
        self.assertEqual(len(middle_pts), 28)
        self.assertEqual(len(thumb_pts), 26)
        self.assertGreaterEqual(len(dominant_pts), 80)
        self.assertEqual(len(dominant_pts), 89)

        # 4. Check segments
        self.assertTrue(any(p.segment == 'tip_pad' for p in index_pts))
        self.assertTrue(any('margin' in p.segment for p in index_pts))

    def test_connections_validity(self):
        """Verifies that all skeletal connections connect valid point IDs."""
        points = self.tracker.generate_dense_mesh(self.hand)
        connections = self.tracker.connections
        
        self.assertGreater(len(connections), 100)
        for start_id, end_id in connections:
            self.assertGreaterEqual(start_id, 0)
            self.assertLess(start_id, len(points))
            self.assertGreaterEqual(end_id, 0)
            self.assertLess(end_id, len(points))
            self.assertNotEqual(start_id, end_id)

    def test_resting_detection_under_noisy_camera(self):
        """Verifies that bad camera sensor noise does not trigger false movement."""
        random.seed(1337)
        tracker = DenseHandTracker(sensitivity=1.0)

        # Feed 20 frames of stationary hand with random Gaussian sensor noise
        for i in range(20):
            noisy_hand = make_hand()
            for p in noisy_hand:
                p.x += random.gauss(0, 0.0035)
                p.y += random.gauss(0, 0.0035)
            summary = tracker.process(noisy_hand, now=1000 + i * 0.033)

        # Hand must be recognized as resting
        self.assertTrue(summary['is_resting'])
        self.assertIsNone(summary['micro_intent'])
        self.assertIsNone(summary['micro_gesture'])
        # Noise floor was automatically learned and adapted
        self.assertGreater(summary['noise_floor'], 0.001)

    def test_slightest_movement_toward_top(self):
        """Verifies that subtle upward deflection of the index finger is detected as UP."""
        tracker = DenseHandTracker(sensitivity=1.0)
        # Settle
        tracker.process(make_hand(), now=1000.0)

        fired_gesture = None
        detected_up = False

        for i in range(12):
            # Move index finger slightly upward (-y)
            dy = -0.0018 * (i + 1)
            hand = make_hand(fingers={8: (0, dy), 7: (0, dy * 0.85), 6: (0, dy * 0.70)})
            summary = tracker.process(hand, now=1000.033 + i * 0.033)

            if summary['micro_intent'] == 'UP':
                detected_up = True
            if summary.get('micro_gesture'):
                fired_gesture = summary['micro_gesture']
                break

        self.assertTrue(detected_up)
        self.assertIsNotNone(fired_gesture)
        self.assertEqual(fired_gesture['type'], 'SWIPE_UP')
        self.assertEqual(fired_gesture['channel'], 'index')
        self.assertGreaterEqual(fired_gesture['confidence'], 0.85)

    def test_slightest_movement_toward_bottom(self):
        """Verifies that subtle downward deflection of the index finger is detected as DOWN."""
        tracker = DenseHandTracker(sensitivity=1.0)
        tracker.process(make_hand(), now=1000.0)

        fired_gesture = None
        detected_down = False

        for i in range(12):
            # Move index finger slightly downward (+y)
            dy = 0.0018 * (i + 1)
            hand = make_hand(fingers={8: (0, dy), 7: (0, dy * 0.85), 6: (0, dy * 0.70)})
            summary = tracker.process(hand, now=1000.033 + i * 0.033)

            if summary['micro_intent'] == 'DOWN':
                detected_down = True
            if summary.get('micro_gesture'):
                fired_gesture = summary['micro_gesture']
                break

        self.assertTrue(detected_down)
        self.assertIsNotNone(fired_gesture)
        self.assertEqual(fired_gesture['type'], 'SWIPE_DOWN')
        self.assertEqual(fired_gesture['channel'], 'index')
        self.assertGreaterEqual(fired_gesture['confidence'], 0.85)

    def test_middle_finger_micro_flick(self):
        """Verifies that a micro flick of the co-dominant middle finger is detected."""
        tracker = DenseHandTracker(sensitivity=1.0)
        tracker.process(make_hand(), now=1000.0)

        fired_gesture = None
        for i in range(12):
            dy = -0.0020 * (i + 1)
            hand = make_hand(fingers={12: (0, dy), 11: (0, dy * 0.85), 10: (0, dy * 0.70)})
            summary = tracker.process(hand, now=1000.033 + i * 0.033)
            if summary.get('micro_gesture'):
                fired_gesture = summary['micro_gesture']
                break

        self.assertIsNotNone(fired_gesture)
        self.assertEqual(fired_gesture['type'], 'SWIPE_UP')
        self.assertEqual(fired_gesture['channel'], 'middle')

    def test_optical_flow_refinement_execution(self):
        """Verifies optical flow refinement executes without error on RGB frames."""
        tracker = DenseHandTracker(enable_optical_flow=True)
        frame1 = np.full((480, 640, 3), 120, dtype=np.uint8)
        frame2 = np.full((480, 640, 3), 125, dtype=np.uint8)

        # Draw a synthetic patch to produce real gradients
        cv2.circle(frame1, (300, 200), 25, (220, 220, 220), -1)
        cv2.circle(frame2, (300, 195), 25, (220, 220, 220), -1)

        summary1 = tracker.process(make_hand(), frame_rgb=frame1, now=1000.0)
        summary2 = tracker.process(make_hand(), frame_rgb=frame2, now=1000.033)
        self.assertEqual(summary2['total_points'], 134)


if __name__ == '__main__':
    unittest.main()
