"""
Unit tests for the Standardized UP and DOWN Gesture Dataset & Kinematic Library.
"""

import json
import os
import tempfile
import unittest

from app.gesture_dataset import (
    get_gesture_dataset,
    export_dataset_json,
    build_gesture_references_from_dataset,
    sync_default_references,
    sync_user_profile,
    _generate_minimum_jerk_trajectory,
    _make_standard_landmark_template,
)


class GestureDatasetTests(unittest.TestCase):
    """Test structured dataset partitioned into UP and DOWN gestures."""

    def test_dataset_structure_has_up_and_down(self):
        dataset = get_gesture_dataset()
        self.assertIn("UP", dataset)
        self.assertIn("DOWN", dataset)
        self.assertIn("rest_baseline", dataset)
        self.assertTrue(dataset.get("is_valid"))

        # Verify UP partition
        up = dataset["UP"]
        self.assertEqual(up["direction"], "UP")
        self.assertGreater(len(up["archetypes"]), 0)
        self.assertEqual(up["optimal_threshold"], 0.022)

        # Verify DOWN partition
        down = dataset["DOWN"]
        self.assertEqual(down["direction"], "DOWN")
        self.assertGreater(len(down["archetypes"]), 0)
        self.assertEqual(down["optimal_threshold"], 0.022)

    def test_archetype_categories(self):
        dataset = get_gesture_dataset()
        up_ids = [m["id"] for m in dataset["UP"]["archetypes"]]
        down_ids = [m["id"] for m in dataset["DOWN"]["archetypes"]]

        # Both UP and DOWN must cover Micro, Natural, Full, and Touch Contact
        self.assertIn("UP_1", up_ids)
        self.assertIn("UP_2", up_ids)
        self.assertIn("UP_3", up_ids)
        self.assertIn("UP_TOUCH", up_ids)

        self.assertIn("DOWN_1", down_ids)
        self.assertIn("DOWN_2", down_ids)
        self.assertIn("DOWN_3", down_ids)
        self.assertIn("DOWN_TOUCH", down_ids)

    def test_minimum_jerk_trajectory_kinematics(self):
        """Minimum-jerk trajectories should have smooth bell-shaped velocity."""
        traj_up = _generate_minimum_jerk_trajectory("UP", peak_disp=0.065, peak_vel=1.25, duration=0.20, num_samples=16)
        self.assertEqual(len(traj_up), 16)
        self.assertEqual(traj_up[0]["disp"], 0.0)
        self.assertEqual(traj_up[0]["vy"], 0.0)
        # End displacement reaches peak
        self.assertAlmostEqual(abs(traj_up[-1]["disp"]), 0.065, places=4)
        # Displacement for UP must be negative (screen Y)
        self.assertLess(traj_up[-1]["disp"], 0.0)

        traj_down = _generate_minimum_jerk_trajectory("DOWN", peak_disp=0.065, peak_vel=1.25, duration=0.20, num_samples=16)
        # Displacement for DOWN must be positive (screen Y)
        self.assertGreater(traj_down[-1]["disp"], 0.0)

    def test_landmark_template_dimensions(self):
        for pose in ("UP_EXTENDED", "DOWN_FLEXED", "TOUCH_THUMB_INDEX", "TOUCH_THUMB_MIDDLE", "NEUTRAL"):
            lms = _make_standard_landmark_template(pose)
            self.assertEqual(len(lms), 21)
            for pt in lms:
                self.assertIn("x", pt)
                self.assertIn("y", pt)
                self.assertIn("z", pt)

    def test_build_references_format_conformance(self):
        refs = build_gesture_references_from_dataset()
        self.assertTrue(refs["is_valid"])
        self.assertEqual(len(refs["up_positions"]), 3)
        self.assertEqual(len(refs["down_positions"]), 3)
        self.assertEqual(len(refs["up_trajectories"]), 3)
        self.assertEqual(len(refs["down_trajectories"]), 3)
        self.assertEqual(refs["up_threshold"], 0.022)
        self.assertEqual(refs["down_threshold"], 0.022)

    def test_export_and_sync_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "dataset.json")
            ref_path = os.path.join(tmpdir, "refs.json")
            prof_path = os.path.join(tmpdir, "profile.json")

            export_dataset_json(ds_path)
            sync_default_references(ref_path)
            sync_user_profile(prof_path)

            self.assertTrue(os.path.exists(ds_path))
            self.assertTrue(os.path.exists(ref_path))
            self.assertTrue(os.path.exists(prof_path))

            with open(ds_path, "r", encoding="utf-8") as f:
                ds_data = json.load(f)
                self.assertIn("UP", ds_data)
                self.assertIn("DOWN", ds_data)

            with open(ref_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
                self.assertEqual(ref_data["up_threshold"], 0.022)
                self.assertEqual(ref_data["down_threshold"], 0.022)


if __name__ == "__main__":
    unittest.main()
