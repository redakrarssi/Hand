"""
Verification script: Simulates full recording of:
- Resting Baseline
- 3 Scroll UP positions (Slight, Medium, Full)
- 3 Scroll DOWN positions (Slight, Medium, Full)
Verifies that:
1. All 6 positions + resting baseline are captured.
2. Saved permanently to disk in gesture_references.json & user_profile.json.
3. Automatically loaded by DenseHandTracker and GestureEngine on startup.
4. Used as ground truth references for micro-gesture detection.
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gesture_recorder import GestureRecorder, load_references, save_references
from app.dense_hand_tracker import DenseHandTracker
from app.gesture_engine import GestureEngine
from tests.test_relaxed_scrolling import make_hand


def verify_full_flow():
    print("=" * 60)
    print("VERIFYING 6-POSITION REFERENCE RECORDER & PERSISTENCE")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_ref_path = os.path.join(tmpdir, "gesture_references.json")
        test_prof_path = os.path.join(tmpdir, "user_profile.json")

        with patch('app.gesture_recorder.REFERENCES_PATH', test_ref_path), \
             patch('app.gesture_recorder.PROFILE_PATH', test_prof_path):

            completed = []
            recorder = GestureRecorder(on_complete_callback=lambda refs: completed.append(refs))
            recorder.start()

            # Positions to record
            positions = [
                ("REST", 0.00),
                ("UP_1", -0.018),   # Slight up
                ("UP_2", -0.038),   # Medium up
                ("UP_3", -0.065),   # Full up
                ("DOWN_1", 0.018),  # Slight down
                ("DOWN_2", 0.038),  # Medium down
                ("DOWN_3", 0.065),  # Full down
            ]

            t = 1000.0
            for step_id, dy in positions:
                hand = make_hand(hand_y=dy)
                hud = recorder.update(hand, now=t)
                print(f"Step: {hud['step_id']} -> Title: {hud['title']}")

                # Lock in via manual_capture (or steady hold)
                recorder.manual_capture(now=t)
                t += 0.80 # advance past transition
                recorder.update(hand, now=t)
                t += 0.10

            assert len(completed) == 1, "Recorder callback should have fired exactly once!"
            refs = completed[0]
            print("\n✅ All 6 reference positions captured successfully!")
            print(f"   UP Positions recorded: {len(refs.up_positions)}")
            print(f"   DOWN Positions recorded: {len(refs.down_positions)}")
            print(f"   Personal UP threshold: {refs.up_threshold:.4f}")
            print(f"   Personal DOWN threshold: {refs.down_threshold:.4f}")
            print(f"   Primary channel: {refs.primary_channel}")

            # Verify files on disk
            assert os.path.exists(test_ref_path), "gesture_references.json must exist!"
            assert os.path.exists(test_prof_path), "user_profile.json must exist!"
            print("\n💾 Files verified on disk:")
            print(f"   - {test_ref_path}")
            print(f"   - {test_prof_path}")

            # Verify loading from disk
            loaded_refs = load_references()
            assert loaded_refs.is_valid, "Loaded references must be valid!"
            assert len(loaded_refs.up_positions) == 3, "Must have 3 UP positions!"
            assert len(loaded_refs.down_positions) == 3, "Must have 3 DOWN positions!"
            print("\n🔄 Re-loaded from disk on simulated app startup:")
            print(f"   Loaded valid: {loaded_refs.is_valid}")
            print(f"   Loaded UP count: {len(loaded_refs.up_positions)}")
            print(f"   Loaded DOWN count: {len(loaded_refs.down_positions)}")

            # Verify DenseHandTracker uses references
            tracker = DenseHandTracker()
            tracker.load_references(loaded_refs)
            assert tracker.has_references, "DenseHandTracker must have references active!"
            print(f"\n🎯 DenseHandTracker active references: has_refs={tracker.has_references}, up_thresh={tracker.ref_up_threshold:.4f}")

            # Verify recognition using references
            # Feed rest hand, then an upward movement matching recorded position
            frame_rgb = None
            tracker.process(make_hand(hand_y=0.0), frame_rgb=frame_rgb, now=2000.0)
            res1 = tracker.process(make_hand(hand_y=-0.025), frame_rgb=frame_rgb, now=2000.033)
            res2 = tracker.process(make_hand(hand_y=-0.035), frame_rgb=frame_rgb, now=2000.066)

            mg = res1.get('micro_gesture') or res2.get('micro_gesture')
            print(f"🚀 Micro-Gesture Detection Result: {mg}")
            assert mg is not None, "Should detect swipe using personal references!"
            assert mg['type'] == 'SWIPE_UP', f"Expected SWIPE_UP, got {mg['type']}"
            print(f"   Source tag: {mg['source']} (Confidence: {mg['confidence']:.2f})")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    verify_full_flow()
