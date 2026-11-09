"""
Analyze Recording Tool - Inspects hand movement session datasets and reports kinematics.

Usage:
    python analyze_recording.py [optional_path_to_session.json] [--apply]
"""

import sys
import os
import json
import argparse

# Add HandScroller root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from app.kinematics_analyzer import KinematicsAnalyzer

RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "recordings")
REFERENCES_PATH = os.path.join(SCRIPT_DIR, "gesture_references.json")
PROFILE_PATH = os.path.join(SCRIPT_DIR, "user_profile.json")


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded hand movement kinematics")
    parser.add_argument('file', nargs='?', default=None, help='Path to session JSON file (defaults to latest_session.json)')
    parser.add_argument('--apply', action='store_true', help='Apply recommended settings directly to profile & references')
    args = parser.parse_args()

    session_file = args.file
    if not session_file:
        session_file = os.path.join(RECORDINGS_DIR, "latest_session.json")

    if not os.path.exists(session_file):
        print(f"❌ Error: No session file found at {session_file}")
        print("Run the training station first: python training_station.py")
        sys.exit(1)

    print(f"🔍 Analyzing session dataset: {session_file} ...\n")
    with open(session_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    profile = KinematicsAnalyzer.analyze_session_data(data)
    report_md = KinematicsAnalyzer.generate_markdown_report(profile)
    print(report_md)

    if args.apply:
        success = KinematicsAnalyzer.apply_profile_to_disk(profile, PROFILE_PATH, REFERENCES_PATH)
        if success:
            print("\n✅ Successfully applied calibrated thresholds to gesture_references.json and user_profile.json!")
        else:
            print("\n❌ Failed to apply parameters to disk.")


if __name__ == "__main__":
    main()
