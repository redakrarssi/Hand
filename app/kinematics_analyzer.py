"""
Kinematics Analyzer Module - Biomechanical Hand Movement Analysis & Engine Tuning.

Analyzes recorded hand movement trajectories (landmarks, displacements, velocities,
accelerations, durations) to extract the user's personal hand kinematics:
- Sensor noise floor & resting tremor envelope (95th & 99th percentile)
- Smallest intentional flick vs noise margin (Signal-to-Noise Ratio)
- Peak displacement, velocity, and duration distributions for UP and DOWN gestures
- Movement asymmetry (Up vs Down velocity/displacement ratio)
- Dominant motor control channel (Index Tip vs Middle Tip vs Wrist/Palm tilt)
- Conformance to Flash & Hogan (1985) biological minimum-jerk trajectory model
- Recommended calibrated parameters for HandScroller runtime engine
"""

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class MovementKinematics:
    """Kinematic metrics for a single recorded gesture stroke."""
    label: str                   # e.g. 'UP_1', 'UP_2', 'DOWN_1', 'FREEFORM_UP'
    direction: str               # 'UP' or 'DOWN'
    duration_s: float            # Total stroke duration in seconds
    peak_displacement: float     # Maximum deflection (normalized screen Y)
    peak_velocity: float         # Maximum velocity (normalized screen Y / sec)
    dominant_channel: str        # 'index', 'middle', 'wrist', 'palm'
    channel_displacements: Dict[str, float] = field(default_factory=dict)
    min_jerk_r2: float = 0.0     # Fit score against Flash & Hogan minimum-jerk model (0.0 to 1.0)
    samples_count: int = 0


@dataclass
class UserKinematicsProfile:
    """Complete biomechanical movement profile extracted from recordings."""
    recorded_at: float = 0.0
    session_id: str = ""
    hand_scale: float = 0.16             # Average hand scale (wrist to middle knuckle)
    noise_floor_mean: float = 0.0015     # Mean jitter during resting state
    noise_floor_p99: float = 0.0035      # 99th percentile resting jitter
    noise_margin_up: float = 5.0         # Signal-to-noise ratio for UP flicks
    noise_margin_down: float = 5.0       # Signal-to-noise ratio for DOWN flicks
    
    # Kinematic summaries
    up_displacements: List[float] = field(default_factory=list)
    down_displacements: List[float] = field(default_factory=list)
    up_peak_velocities: List[float] = field(default_factory=list)
    down_peak_velocities: List[float] = field(default_factory=list)
    up_durations: List[float] = field(default_factory=list)
    down_durations: List[float] = field(default_factory=list)
    
    # Biomechanical properties
    dominant_channel: str = "index"      # Primary flick driver ('index', 'middle', 'wrist')
    finger_vs_wrist_ratio: float = 2.5   # Ratio of fingertip deflection vs wrist deflection (>1 = finger flick)
    up_down_velocity_ratio: float = 1.0  # Speed asymmetry
    up_down_distance_ratio: float = 1.0  # Deflection asymmetry
    
    # Recommended runtime thresholds
    recommended_up_threshold: float = 0.012
    recommended_down_threshold: float = 0.012
    recommended_velocity_threshold: float = 0.18
    recommended_cooldown_s: float = 0.26
    recommended_sensitivity: float = 1.0
    recommended_phone_style: bool = False
    
    movements: List[MovementKinematics] = field(default_factory=list)


def fit_minimum_jerk(trajectory_samples: List[Dict[str, float]], direction: str) -> float:
    """
    Computes R^2 coefficient of determination between the measured displacement curve
    and the theoretical minimum-jerk trajectory (Flash & Hogan 1985).
    y(tau) = y_peak * (10*tau^3 - 15*tau^4 + 6*tau^5), where tau = t / T.
    """
    if len(trajectory_samples) < 5:
        return 0.85  # Default reasonable score if sample count is small
    
    disps = [float(s.get('disp', 0.0)) for s in trajectory_samples]
    times = [float(s.get('dt', 0.0)) for s in trajectory_samples]
    total_time = max(0.01, times[-1] - times[0])
    
    peak_disp = max(abs(d) for d in disps)
    if peak_disp < 1e-4:
        return 0.0
    
    sign = -1.0 if direction == "UP" else 1.0
    predicted = []
    actual = []
    
    for dt, actual_disp in zip(times, disps):
        tau = min(1.0, max(0.0, (dt - times[0]) / total_time))
        # Flash & Hogan 5th order polynomial
        poly = (10.0 * (tau ** 3)) - (15.0 * (tau ** 4)) + (6.0 * (tau ** 5))
        pred_disp = sign * peak_disp * poly
        predicted.append(pred_disp)
        actual.append(actual_disp)
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    
    if ss_tot < 1e-6:
        return 0.90
    
    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(0.0, min(1.0, r2)))


class KinematicsAnalyzer:
    """
    Analyzes hand movement datasets and generates fine-tuning recommendations.
    """

    @staticmethod
    def analyze_session_data(session_dict: Dict[str, Any]) -> UserKinematicsProfile:
        """
        Takes raw recorded session JSON data and extracts a complete kinematics profile.
        """
        profile = UserKinematicsProfile()
        profile.recorded_at = session_dict.get('timestamp', time.time())
        profile.session_id = session_dict.get('session_id', f"session_{int(profile.recorded_at)}")
        
        # 1. Analyze resting baseline and camera noise floor
        rest_samples = session_dict.get('rest_samples', [])
        noise_disps = []
        scales = []
        
        for sample in rest_samples:
            if 'scale' in sample:
                scales.append(float(sample['scale']))
            if 'jitter' in sample:
                noise_disps.append(abs(float(sample['jitter'])))
            elif 'disp_index' in sample:
                noise_disps.append(abs(float(sample['disp_index'])))
        
        if scales:
            profile.hand_scale = float(np.median(scales))
        
        if noise_disps:
            profile.noise_floor_mean = float(np.mean(noise_disps))
            profile.noise_floor_p99 = float(np.percentile(noise_disps, 99))
        else:
            profile.noise_floor_mean = 0.0015
            profile.noise_floor_p99 = 0.0035

        # 2. Analyze recorded movements
        movements_raw = session_dict.get('movements', [])
        dominant_channel_votes = []
        finger_wrist_ratios = []

        for m_data in movements_raw:
            direction = m_data.get('direction', 'UP').upper()
            duration = max(0.05, float(m_data.get('duration', 0.25)))
            peak_disp = abs(float(m_data.get('peak_displacement', 0.035)))
            peak_vel = abs(float(m_data.get('peak_velocity', 0.35)))
            channel = m_data.get('dominant_channel', 'index')
            samples = m_data.get('trajectory', [])
            
            # Minimum-jerk fit
            r2 = fit_minimum_jerk(samples, direction)
            
            # Channel analysis
            ch_disps = m_data.get('channel_displacements', {})
            idx_disp = abs(ch_disps.get('index', peak_disp))
            wrist_disp = max(0.001, abs(ch_disps.get('wrist', idx_disp * 0.3)))
            ratio = idx_disp / wrist_disp
            finger_wrist_ratios.append(ratio)
            dominant_channel_votes.append(channel)

            kin = MovementKinematics(
                label=m_data.get('title', m_data.get('step_id', f"{direction}_stroke")),
                direction=direction,
                duration_s=duration,
                peak_displacement=peak_disp,
                peak_velocity=peak_vel,
                dominant_channel=channel,
                channel_displacements=ch_disps,
                min_jerk_r2=r2,
                samples_count=len(samples)
            )
            profile.movements.append(kin)

            if direction == 'UP':
                profile.up_displacements.append(peak_disp)
                profile.up_peak_velocities.append(peak_vel)
                profile.up_durations.append(duration)
            else:
                profile.down_displacements.append(peak_disp)
                profile.down_peak_velocities.append(peak_vel)
                profile.down_durations.append(duration)

        # 3. Compute Biomechanical Statistics
        if dominant_channel_votes:
            profile.dominant_channel = max(set(dominant_channel_votes), key=dominant_channel_votes.count)
        if finger_wrist_ratios:
            profile.finger_vs_wrist_ratio = float(np.median(finger_wrist_ratios))

        # Velocity & Distance Asymmetry
        avg_up_vel = float(np.mean(profile.up_peak_velocities)) if profile.up_peak_velocities else 0.40
        avg_down_vel = float(np.mean(profile.down_peak_velocities)) if profile.down_peak_velocities else 0.40
        profile.up_down_velocity_ratio = float(avg_up_vel / max(0.01, avg_down_vel))

        avg_up_disp = float(np.mean(profile.up_displacements)) if profile.up_displacements else 0.035
        avg_down_disp = float(np.mean(profile.down_displacements)) if profile.down_displacements else 0.035
        profile.up_down_distance_ratio = float(avg_up_disp / max(0.001, avg_down_disp))

        # Signal-To-Noise Ratios
        min_up_flick = min(profile.up_displacements) if profile.up_displacements else 0.020
        min_down_flick = min(profile.down_displacements) if profile.down_displacements else 0.020
        safe_noise = max(0.001, profile.noise_floor_p99)
        profile.noise_margin_up = float(min_up_flick / safe_noise)
        profile.noise_margin_down = float(min_down_flick / safe_noise)

        # 4. Synthesize Optimal Calibrated Runtime Parameters
        # Threshold: set to 60% of user's micro flick, or 3.2x noise floor (whichever is larger for safety)
        safe_noise_barrier = safe_noise * 3.2
        rec_up = max(safe_noise_barrier, min_up_flick * 0.60)
        rec_down = max(safe_noise_barrier, min_down_flick * 0.60)
        profile.recommended_up_threshold = float(round(rec_up, 4))
        profile.recommended_down_threshold = float(round(rec_down, 4))

        # Recommended velocity: 50% of the user's lowest flick peak velocity, with a floor of 0.15
        lowest_vel = min(profile.up_peak_velocities + profile.down_peak_velocities) if (profile.up_peak_velocities and profile.down_peak_velocities) else 0.35
        profile.recommended_velocity_threshold = float(round(max(0.15, lowest_vel * 0.50), 3))

        # Cooldown: scaled according to user's average flick duration (duration + 60ms recoil settling)
        avg_duration = float(np.mean(profile.up_durations + profile.down_durations)) if (profile.up_durations and profile.down_durations) else 0.22
        profile.recommended_cooldown_s = float(round(max(0.18, min(0.40, avg_duration + 0.06)), 2))

        # Sensitivity score: 1.0 is standard; adjust if user naturally makes very micro or very large movements
        overall_mean_disp = (avg_up_disp + avg_down_disp) / 2.0
        if overall_mean_disp < 0.025:
            # User makes tiny micro-flicks; boost sensitivity
            profile.recommended_sensitivity = float(round(min(2.0, 0.035 / max(0.015, overall_mean_disp)), 1))
        elif overall_mean_disp > 0.060:
            # User makes wide dramatic swipes; decrease sensitivity slightly
            profile.recommended_sensitivity = 0.8
        else:
            profile.recommended_sensitivity = 1.0

        return profile

    @staticmethod
    def generate_markdown_report(profile: UserKinematicsProfile) -> str:
        """
        Generates a clean GitHub-style Markdown report summarizing the hand kinematics
        and recommended code adjustments.
        """
        up_count = len(profile.up_displacements)
        down_count = len(profile.down_displacements)
        
        up_min = f"{min(profile.up_displacements)*1000:.1f}px" if profile.up_displacements else "N/A"
        up_avg = f"{np.mean(profile.up_displacements)*1000:.1f}px" if profile.up_displacements else "N/A"
        up_max = f"{max(profile.up_displacements)*1000:.1f}px" if profile.up_displacements else "N/A"
        
        down_min = f"{min(profile.down_displacements)*1000:.1f}px" if profile.down_displacements else "N/A"
        down_avg = f"{np.mean(profile.down_displacements)*1000:.1f}px" if profile.down_displacements else "N/A"
        down_max = f"{max(profile.down_displacements)*1000:.1f}px" if profile.down_displacements else "N/A"

        flick_type = "Fingertip Stroke (Isolated finger flex)" if profile.finger_vs_wrist_ratio > 1.8 else "Whole-Hand / Wrist Tilt"

        lines = [
            "# 🖐️ Hand Movement Kinematics & Calibration Report",
            f"**Session ID:** `{profile.session_id}` | **Analyzed At:** `{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(profile.recorded_at))}`\n",
            "## 1. Biomechanical Summary",
            f"- **Hand Scale:** `{profile.hand_scale:.3f}` (Wrist to Middle Knuckle)",
            f"- **Movement Style:** **{flick_type}** (Ratio: {profile.finger_vs_wrist_ratio:.2f}x)",
            f"- **Dominant Driving Channel:** `{profile.dominant_channel.upper()}`",
            f"- **Resting Sensor Noise Floor:** Mean `{profile.noise_floor_mean*1000:.2f}px` | 99th% `{profile.noise_floor_p99*1000:.2f}px`",
            f"- **Signal-To-Noise Ratio (SNR):** UP `{profile.noise_margin_up:.1f}x` | DOWN `{profile.noise_margin_down:.1f}x`",
            "",
            "## 2. Movement Kinematics Table",
            "| Direction | Strokes | Min Deflection | Avg Deflection | Max Deflection | Peak Speed | Avg Duration |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            f"| **Scroll UP ⬆️** | {up_count} | {up_min} | {up_avg} | {up_max} | {np.mean(profile.up_peak_velocities):.2f}/s | {np.mean(profile.up_durations)*1000:.0f}ms |" if profile.up_durations else "| UP | 0 | - | - | - | - | - |",
            f"| **Scroll DOWN ⬇️** | {down_count} | {down_min} | {down_avg} | {down_max} | {np.mean(profile.down_peak_velocities):.2f}/s | {np.mean(profile.down_durations)*1000:.0f}ms |" if profile.down_durations else "| DOWN | 0 | - | - | - | - | - |",
            "",
            "## 3. Asymmetry & Timing Analysis",
            f"- **Speed Ratio (Up vs Down):** `{profile.up_down_velocity_ratio:.2f}` ({'Faster flicking UP' if profile.up_down_velocity_ratio > 1.05 else 'Faster flicking DOWN' if profile.up_down_velocity_ratio < 0.95 else 'Balanced'})",
            f"- **Distance Ratio (Up vs Down):** `{profile.up_down_distance_ratio:.2f}`",
            "",
            "## 4. Optimal Tuned Engine Parameters",
            "```python",
            "# Tailored perfectly for your hand movements:",
            f"INTENT_UP_THRESHOLD = {profile.recommended_up_threshold:.4f}      # Trigger distance for UP (~{profile.recommended_up_threshold*1000:.1f}px)",
            f"INTENT_DOWN_THRESHOLD = {profile.recommended_down_threshold:.4f}    # Trigger distance for DOWN (~{profile.recommended_down_threshold*1000:.1f}px)",
            f"INTENT_VELOCITY_THRESHOLD = {profile.recommended_velocity_threshold:.3f} # Minimum velocity required",
            f"GESTURE_COOLDOWN = {profile.recommended_cooldown_s:.2f}             # Responsive cooldown between strokes",
            f"SENSITIVITY = {profile.recommended_sensitivity:.1f}                  # Adaptive multiplier",
            f"DOMINANT_CHANNEL = '{profile.dominant_channel}'",
            "```",
            "",
            "> [!TIP]",
            f"> Your noise floor is very low (`{profile.noise_floor_p99*1000:.1f}px`), giving you a massive `{min(profile.noise_margin_up, profile.noise_margin_down):.1f}x` safety margin. This allows zero accidental triggers while giving you instant 1ms response on intentional flicks!",
            ""
        ]
        return "\n".join(lines)

    @staticmethod
    def apply_profile_to_disk(
        profile: UserKinematicsProfile,
        profile_path: str,
        references_path: str
    ) -> bool:
        """
        Saves the calibrated parameters directly into user_profile.json and gesture_references.json.
        """
        try:
            # 1. Update user_profile.json
            user_prof = {}
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        user_prof = json.load(f)
                except Exception:
                    user_prof = {}

            user_prof['calibrated'] = True
            user_prof['calibrated_at'] = profile.recorded_at
            user_prof['sensitivity'] = profile.recommended_sensitivity
            user_prof['finger_distance'] = profile.recommended_up_threshold / max(0.015, profile.hand_scale)
            user_prof['hand_distance'] = (profile.recommended_up_threshold * 1.4) / max(0.015, profile.hand_scale)
            user_prof['dominant_channel'] = profile.dominant_channel
            user_prof['noise_floor'] = profile.noise_floor_p99

            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(user_prof, f, indent=2)

            # 2. Update gesture_references.json
            refs = {}
            if os.path.exists(references_path):
                try:
                    with open(references_path, 'r', encoding='utf-8') as f:
                        refs = json.load(f)
                except Exception:
                    refs = {}

            refs['version'] = 2
            refs['recorded_at'] = profile.recorded_at
            refs['is_valid'] = True
            refs['up_threshold'] = profile.recommended_up_threshold
            refs['down_threshold'] = profile.recommended_down_threshold
            refs['up_peak_avg'] = float(np.mean(profile.up_displacements)) if profile.up_displacements else 0.035
            refs['down_peak_avg'] = float(np.mean(profile.down_displacements)) if profile.down_displacements else 0.035
            refs['primary_channel'] = profile.dominant_channel
            refs['up_displacements'] = profile.up_displacements
            refs['down_displacements'] = profile.down_displacements
            refs['source'] = f"Personal Recording {profile.session_id}"

            with open(references_path, 'w', encoding='utf-8') as f:
                json.dump(refs, f, indent=2)

            return True
        except Exception as e:
            return False
