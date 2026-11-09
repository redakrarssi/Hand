"""
Dense Hand Tracker Module - High-Density Anatomical Hand Tracking & Micro-Movement Intent.

Extracts 134 dense anatomical tracking points on the hand (over 6x more points than
standard 21 MediaPipe landmarks), with heavy concentration on dominant fingers (Index: 32,
Middle: 28, Thumb: 26).

Specifically engineered for low-cost, noisy webcams ("bad camera"):
- Uses multi-point directional consensus to eliminate sensor noise (gain of ~5.6x SNR).
- Distinguishes rock-solid hand resting from the slightest micro-movements (top/bottom).
- Subpixel optical flow / gradient tracking for frame-to-frame micro-deflections.
- Adaptive learning of the user's hand resting envelope and specific camera sensor noise floor.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


@dataclass
class DensePoint:
    """A single tracking point on the dense hand model."""
    id: int
    x: float            # Normalized X in [0.0, 1.0]
    y: float            # Normalized Y in [0.0, 1.0]
    z: float            # Normalized depth Z
    region: str         # 'index', 'middle', 'thumb', 'ring', 'pinky', 'palm'
    segment: str        # 'joint', 'proximal', 'intermediate', 'distal', 'pad', 'palm_base'
    is_dominant: bool   # True for index, middle, thumb
    weight: float       # Weight in consensus velocity calculations


class DenseHandTracker:
    """
    Constructs a 134-point dense anatomical mesh and tracks micro-intent
    (resting vs top/bottom scroll intent) with high noise immunity.
    """

    def __init__(self, sensitivity: float = 1.0, enable_optical_flow: bool = True, autoload_references: bool = False):
        self.sensitivity = max(0.3, min(3.0, sensitivity))
        self.enable_optical_flow = enable_optical_flow
        self.autoload_references = autoload_references

        # State tracking
        self.points: List[DensePoint] = []
        self.connections: List[Tuple[int, int]] = []
        self.prev_points: Optional[List[DensePoint]] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_time: float = 0.0

        # Resting & Micro-movement state
        self.is_resting: bool = True
        self.resting_stability: float = 1.0  # 0.0 (erratic) to 1.0 (rock solid)
        self.micro_intent: Optional[str] = None  # 'UP', 'DOWN', or None
        self.micro_velocity: float = 0.0
        self.directional_coherence: float = 0.0

        # Micro-gesture event detection
        self.micro_stroke_accumulator: float = 0.0
        self.micro_stroke_direction: Optional[str] = None
        self.micro_stroke_start_time: float = 0.0
        self.last_micro_fire_time: float = -math.inf
        self.micro_cooldown: float = 0.28  # Seconds between micro-scroll fires
        self.consecutive_flick_frames: int = 0
        self.flick_direction: Optional[str] = None
        self.last_fired_direction: Optional[str] = None

        # Baseline resting reference for dominant fingers
        self.resting_baseline_y: Optional[float] = None
        self.resting_baseline_pts: Optional[Dict[str, float]] = None
        self.channel_baselines: Dict[str, float] = {}
        self.baseline_learning_rate: float = 0.03
        self.noise_floor: float = 0.0018  # Auto-calibrated camera noise floor
        self.noise_samples: deque = deque(maxlen=60)

        # History queues for temporal filtering and coherence
        self.velocity_history = deque(maxlen=15)
        self.displacement_history = deque(maxlen=20)
        self.resting_history = deque(maxlen=30)

        # Statistics
        self.dominant_deflections = {'index': 0.0, 'middle': 0.0, 'thumb': 0.0}
        self.point_count = 134
        self.dominant_point_count = 89  # 86 on dominant fingers + 3 palm/web anchors
        self.finger_point_count = 116    # 32 + 28 + 26 + 16 + 14

        # Personal Recorded Gesture References (3 UP, 3 DOWN whole movements)
        self.references: Optional[Any] = None
        self.has_references: bool = False
        self.ref_up_threshold: float = 0.012
        self.ref_down_threshold: float = 0.012
        self.ref_primary_channel: str = 'index'
        self.reference_up_templates: List[Dict[str, Any]] = []
        self.reference_down_templates: List[Dict[str, Any]] = []
        self.reference_up_trajectories: List[Dict[str, Any]] = []
        self.reference_down_trajectories: List[Dict[str, Any]] = []
        self.up_peak_avg: float = 0.035
        self.down_peak_avg: float = 0.035
        self.last_template_match: Optional[str] = None
        self.last_template_similarity: float = 0.0
        self.last_trajectory_match: Optional[str] = None
        self.last_trajectory_similarity: float = 0.0

        # Auto-load references from disk if explicitly requested
        if self.autoload_references:
            self._try_autoload_references()

    def _try_autoload_references(self):
        """Attempt to load saved reference templates from disk."""
        try:
            from app.gesture_recorder import load_references
            refs = load_references()
            if refs and getattr(refs, 'is_valid', False):
                self.load_references(refs)
        except Exception:
            pass

    def load_references(self, refs):
        """Load and apply personal 6-movement reference templates and trajectories."""
        if not refs or not getattr(refs, 'is_valid', False):
            return
        self.references = refs
        self.has_references = True
        self.ref_up_threshold = max(0.004, min(0.035, float(getattr(refs, 'up_threshold', 0.012))))
        self.ref_down_threshold = max(0.004, min(0.035, float(getattr(refs, 'down_threshold', 0.012))))
        self.ref_primary_channel = getattr(refs, 'primary_channel', 'index')
        self.reference_up_templates = getattr(refs, 'up_positions', []) or []
        self.reference_down_templates = getattr(refs, 'down_positions', []) or []
        self.reference_up_trajectories = getattr(refs, 'up_trajectories', []) or []
        self.reference_down_trajectories = getattr(refs, 'down_trajectories', []) or []
        self.up_peak_avg = max(0.005, min(0.080, float(getattr(refs, 'up_peak_avg', 0.035))))
        self.down_peak_avg = max(0.005, min(0.080, float(getattr(refs, 'down_peak_avg', 0.035))))

    def _compute_trajectory_match(self, current_dir: str, disp: float, vy: float) -> Tuple[float, Optional[str]]:
        """
        Compare active stroke dynamics (displacement & velocity) to the user's
        recorded whole-movement trajectories.
        Returns (highest_conformance_score, matched_movement_id).
        """
        if not self.has_references:
            return 0.0, None

        trajectories = self.reference_up_trajectories if current_dir == 'UP' else self.reference_down_trajectories
        if not trajectories:
            return 0.0, None

        best_score = 0.0
        best_id = None

        abs_disp = abs(disp)
        abs_vy = abs(vy)

        for m in trajectories:
            peak_d = max(0.005, float(m.get('peak_displacement', 0.020)))
            peak_v = max(0.05, float(m.get('peak_velocity', 0.50)))

            # Progress toward user's recorded stroke peak
            r_disp = min(1.0, abs_disp / (peak_d * 0.70))
            # Speed conformance with user's stroke speed
            r_vel = min(1.0, abs_vy / (peak_v * 0.55))

            score = 0.65 * r_disp + 0.35 * r_vel
            if score > best_score:
                best_score = score
                best_id = m.get('step_id')

        return float(best_score), best_id

    def _compute_template_match(self, landmarks, current_dir: str) -> Tuple[float, Optional[str]]:
        """
        Compare current MediaPipe landmarks to the user's recorded reference templates.
        Returns (highest_similarity, matched_step_id).
        """
        if not self.has_references or not landmarks or len(landmarks) < 21:
            return 0.0, None

        templates = self.reference_up_templates if current_dir == 'UP' else self.reference_down_templates
        if not templates:
            return 0.0, None

        wrist = landmarks[0]
        knuckle = landmarks[9]
        scale = max(0.015, math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y))

        # Check key finger landmarks (wrist 0, index 5..8, middle 9..12)
        eval_indices = (0, 5, 6, 7, 8, 9, 10, 11, 12)

        best_sim = 0.0
        best_id = None

        for tpl in templates:
            tpl_lms = tpl.get('landmarks_norm')
            if not tpl_lms or len(tpl_lms) < 21:
                continue

            diffs = []
            for idx in eval_indices:
                curr_nx = (landmarks[idx].x - wrist.x) / scale
                curr_ny = (landmarks[idx].y - wrist.y) / scale
                tpl_nx = tpl_lms[idx]['x']
                tpl_ny = tpl_lms[idx]['y']
                diffs.append(math.hypot(curr_nx - tpl_nx, curr_ny - tpl_ny))

            avg_diff = sum(diffs) / len(diffs)
            sim = max(0.0, min(1.0, 1.0 - avg_diff * 2.0))
            if sim > best_sim:
                best_sim = sim
                best_id = tpl.get('step_id')

        return best_sim, best_id

    def reset(self):
        """Reset internal history and calibration."""
        self.points = []
        self.connections = []
        self.prev_points = None
        self.prev_gray = None
        self.prev_time = 0.0
        self.is_resting = True
        self.resting_stability = 1.0
        self.micro_intent = None
        self.micro_velocity = 0.0
        self.directional_coherence = 0.0
        self.resting_baseline_y = None
        self.resting_baseline_pts = None
        self.channel_baselines.clear()
        self.consecutive_flick_frames = 0
        self.flick_direction = None
        self.last_fired_direction = None
        self.micro_stroke_accumulator = 0.0
        self.micro_stroke_direction = None
        self.last_template_match = None
        self.last_template_similarity = 0.0
        self.velocity_history.clear()
        self.displacement_history.clear()
        self.resting_history.clear()

    def set_sensitivity(self, value: float):
        """Update sensitivity."""
        self.sensitivity = max(0.3, min(3.0, value))

    # ─── Dense Anatomical Point Generation ─────────────────────────────

    @staticmethod
    def _interpolate_segment(pt_a, pt_b, fractions: List[float], lateral_fractions: List[float],
                             width: float, region: str, segment: str, is_dominant: bool,
                             weight: float, start_id: int,
                             connections: List[Tuple[int, int]],
                             parent_joint_id: int) -> Tuple[List[DensePoint], int, int, List[int], List[int]]:
        """
        Interpolates interior points along a bone segment, adding central
        spine points, bilateral contour margins, and skeletal connections.
        """
        points = []
        cur_id = start_id
        dx = pt_b.x - pt_a.x
        dy = pt_b.y - pt_a.y
        dz = getattr(pt_b, 'z', 0.0) - getattr(pt_a, 'z', 0.0)
        length = math.hypot(dx, dy)

        if length > 1e-6:
            nx = -dy / length
            ny = dx / length
        else:
            nx, ny = 0.0, 1.0

        center_ids = []
        # Central spine points
        prev_center_id = parent_joint_id
        for t in fractions:
            cx = pt_a.x + t * dx
            cy = pt_a.y + t * dy
            cz = getattr(pt_a, 'z', 0.0) + t * dz
            points.append(DensePoint(
                id=cur_id, x=cx, y=cy, z=cz,
                region=region, segment=segment,
                is_dominant=is_dominant, weight=weight
            ))
            center_ids.append(cur_id)
            if prev_center_id is not None:
                connections.append((prev_center_id, cur_id))
            prev_center_id = cur_id
            cur_id += 1

        left_ids = []
        right_ids = []
        # Bilateral lateral margin points (left and right contour of phalanx)
        for i, t in enumerate(lateral_fractions):
            cx = pt_a.x + t * dx
            cy = pt_a.y + t * dy
            cz = getattr(pt_a, 'z', 0.0) + t * dz
            half_w = (width * (1.0 - 0.25 * t)) * 0.5

            # Left margin
            left_id = cur_id
            points.append(DensePoint(
                id=left_id, x=cx + nx * half_w, y=cy + ny * half_w, z=cz,
                region=region, segment=f"{segment}_margin",
                is_dominant=is_dominant, weight=weight * 0.9
            ))
            left_ids.append(left_id)
            cur_id += 1

            # Right margin
            right_id = cur_id
            points.append(DensePoint(
                id=right_id, x=cx - nx * half_w, y=cy - ny * half_w, z=cz,
                region=region, segment=f"{segment}_margin",
                is_dominant=is_dominant, weight=weight * 0.9
            ))
            right_ids.append(right_id)
            cur_id += 1

            # Connect lateral margin points along contour
            if i > 0:
                connections.append((left_ids[i - 1], left_id))
                connections.append((right_ids[i - 1], right_id))

            # Connect across rib (left to center to right)
            if i < len(center_ids):
                connections.append((left_id, center_ids[i]))
                connections.append((center_ids[i], right_id))

        last_center_id = prev_center_id
        return points, cur_id, last_center_id, left_ids, right_ids

    @staticmethod
    def _create_fingertip_pad_cluster(tip_pt, dip_pt, scale: float, count: int,
                                      region: str, start_id: int,
                                      connections: List[Tuple[int, int]],
                                      tip_id: int) -> Tuple[List[DensePoint], int]:
        """
        Generates a dense radial cluster around the tactile fingertip pad.
        """
        points = []
        cur_id = start_id
        dx = tip_pt.x - dip_pt.x
        dy = tip_pt.y - dip_pt.y
        dz = getattr(tip_pt, 'z', 0.0) - getattr(dip_pt, 'z', 0.0)
        length = math.hypot(dx, dy)

        if length > 1e-6:
            tx, ty = dx / length, dy / length
            nx, ny = -ty, tx
        else:
            tx, ty = 0.0, -1.0
            nx, ny = 1.0, 0.0

        pad_radius = scale * 0.038

        if count == 4:
            angles = [-0.75, -0.25, 0.25, 0.75]
        elif count == 3:
            angles = [-0.55, 0.0, 0.55]
        else:
            angles = [-0.40, 0.40]

        pad_ids = []
        for angle in angles:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            vx = cos_a * tx + sin_a * nx
            vy = cos_a * ty + sin_a * ny

            px = tip_pt.x + vx * pad_radius
            py = tip_pt.y + vy * pad_radius
            pz = getattr(tip_pt, 'z', 0.0) + dz * 0.15

            points.append(DensePoint(
                id=cur_id, x=px, y=py, z=pz,
                region=region, segment='tip_pad',
                is_dominant=True if region in ('index', 'middle', 'thumb') else False,
                weight=1.25
            ))
            connections.append((tip_id, cur_id))
            pad_ids.append(cur_id)
            cur_id += 1

        # Connect pad points in an arc
        for i in range(len(pad_ids) - 1):
            connections.append((pad_ids[i], pad_ids[i + 1]))

        return points, cur_id

    def generate_dense_mesh(self, landmarks) -> List[DensePoint]:
        """
        Builds the 134-point anatomical dense hand mesh from 21 MediaPipe base landmarks.
        Concentrates 86 points on dominant fingers (Index: 32, Middle: 28, Thumb: 26).
        """
        if not landmarks or len(landmarks) != 21:
            return []

        pts = landmarks
        wrist = pts[0]
        scale = math.hypot(pts[9].x - wrist.x, pts[9].y - wrist.y)
        if scale < 0.01:
            scale = 0.15

        dense_points: List[DensePoint] = []
        connections: List[Tuple[int, int]] = []
        cur_id = 0

        def add_joint(idx, region, weight=1.0, dominant=False):
            nonlocal cur_id
            p = pts[idx]
            pt = DensePoint(
                id=cur_id, x=p.x, y=p.y, z=getattr(p, 'z', 0.0),
                region=region, segment='joint',
                is_dominant=dominant, weight=weight
            )
            dense_points.append(pt)
            joint_id = cur_id
            cur_id += 1
            return joint_id

        # ── 1. INDEX FINGER (Dominant - 32 Points) ──
        idx_mcp = add_joint(5, 'index', weight=1.0, dominant=True)
        p_prox, cur_id, last_c1, l1, r1 = self._interpolate_segment(
            pts[5], pts[6], [0.25, 0.50, 0.75], [0.25, 0.50, 0.75],
            scale * 0.11, 'index', 'proximal', True, 1.0, cur_id,
            connections, idx_mcp
        )
        dense_points.extend(p_prox)

        idx_pip = add_joint(6, 'index', weight=1.05, dominant=True)
        connections.append((last_c1, idx_pip))
        p_inter, cur_id, last_c2, l2, r2 = self._interpolate_segment(
            pts[6], pts[7], [0.25, 0.50, 0.75], [0.25, 0.50, 0.75],
            scale * 0.095, 'index', 'intermediate', True, 1.1, cur_id,
            connections, idx_pip
        )
        dense_points.extend(p_inter)

        idx_dip = add_joint(7, 'index', weight=1.1, dominant=True)
        connections.append((last_c2, idx_dip))
        p_dist, cur_id, last_c3, l3, r3 = self._interpolate_segment(
            pts[7], pts[8], [0.35, 0.70], [0.35, 0.70],
            scale * 0.08, 'index', 'distal', True, 1.2, cur_id,
            connections, idx_dip
        )
        dense_points.extend(p_dist)

        idx_tip = add_joint(8, 'index', weight=1.3, dominant=True)
        connections.append((last_c3, idx_tip))
        p_pad, cur_id = self._create_fingertip_pad_cluster(pts[8], pts[7], scale, 4, 'index', cur_id, connections, idx_tip)
        dense_points.extend(p_pad)

        # ── 2. MIDDLE FINGER (Dominant - 28 Points) ──
        mid_mcp = add_joint(9, 'middle', weight=1.0, dominant=True)
        p_prox_m, cur_id, mc1, ml1, mr1 = self._interpolate_segment(
            pts[9], pts[10], [0.33, 0.67], [0.33, 0.67],
            scale * 0.115, 'middle', 'proximal', True, 1.0, cur_id,
            connections, mid_mcp
        )
        dense_points.extend(p_prox_m)

        mid_pip = add_joint(10, 'middle', weight=1.05, dominant=True)
        connections.append((mc1, mid_pip))
        p_inter_m, cur_id, mc2, ml2, mr2 = self._interpolate_segment(
            pts[10], pts[11], [0.25, 0.50, 0.75], [0.25, 0.50, 0.75],
            scale * 0.10, 'middle', 'intermediate', True, 1.1, cur_id,
            connections, mid_pip
        )
        dense_points.extend(p_inter_m)

        mid_dip = add_joint(11, 'middle', weight=1.1, dominant=True)
        connections.append((mc2, mid_dip))
        p_dist_m, cur_id, mc3, ml3, mr3 = self._interpolate_segment(
            pts[11], pts[12], [0.35, 0.70], [0.35, 0.70],
            scale * 0.085, 'middle', 'distal', True, 1.2, cur_id,
            connections, mid_dip
        )
        dense_points.extend(p_dist_m)

        mid_tip = add_joint(12, 'middle', weight=1.3, dominant=True)
        connections.append((mc3, mid_tip))
        p_pad_m, cur_id = self._create_fingertip_pad_cluster(pts[12], pts[11], scale, 3, 'middle', cur_id, connections, mid_tip)
        dense_points.extend(p_pad_m)

        # ── 3. THUMB (Dominant - 26 Points) ──
        th_cmc = add_joint(1, 'thumb', weight=0.9, dominant=True)
        p_t1, cur_id, tc1, _, _ = self._interpolate_segment(
            pts[1], pts[2], [0.35, 0.70], [0.50],
            scale * 0.13, 'thumb', 'cmc_mcp', True, 0.95, cur_id,
            connections, th_cmc
        )
        dense_points.extend(p_t1)

        th_mcp = add_joint(2, 'thumb', weight=1.0, dominant=True)
        connections.append((tc1, th_mcp))
        p_t2, cur_id, tc2, _, _ = self._interpolate_segment(
            pts[2], pts[3], [0.25, 0.50, 0.75], [0.33, 0.67],
            scale * 0.12, 'thumb', 'proximal', True, 1.05, cur_id,
            connections, th_mcp
        )
        dense_points.extend(p_t2)

        th_ip = add_joint(3, 'thumb', weight=1.1, dominant=True)
        connections.append((tc2, th_ip))
        p_t3, cur_id, tc3, _, _ = self._interpolate_segment(
            pts[3], pts[4], [0.25, 0.50, 0.75], [0.33, 0.67],
            scale * 0.10, 'thumb', 'distal', True, 1.15, cur_id,
            connections, th_ip
        )
        dense_points.extend(p_t3)

        th_tip = add_joint(4, 'thumb', weight=1.3, dominant=True)
        connections.append((tc3, th_tip))
        p_pad_t, cur_id = self._create_fingertip_pad_cluster(pts[4], pts[3], scale, 4, 'thumb', cur_id, connections, th_tip)
        dense_points.extend(p_pad_t)

        # ── 4. RING FINGER (16 Points) ──
        rng_mcp = add_joint(13, 'ring', weight=0.85, dominant=False)
        p_r1, cur_id, rc1, _, _ = self._interpolate_segment(
            pts[13], pts[14], [0.35, 0.70], [0.50],
            scale * 0.09, 'ring', 'proximal', False, 0.85, cur_id,
            connections, rng_mcp
        )
        dense_points.extend(p_r1)

        rng_pip = add_joint(14, 'ring', weight=0.9, dominant=False)
        connections.append((rc1, rng_pip))
        p_r2, cur_id, rc2, _, _ = self._interpolate_segment(
            pts[14], pts[15], [0.35, 0.70], [0.50],
            scale * 0.08, 'ring', 'intermediate', False, 0.9, cur_id,
            connections, rng_pip
        )
        dense_points.extend(p_r2)

        rng_dip = add_joint(15, 'ring', weight=0.95, dominant=False)
        connections.append((rc2, rng_dip))
        p_r3, cur_id, rc3, _, _ = self._interpolate_segment(
            pts[15], pts[16], [0.35, 0.70], [],
            scale * 0.07, 'ring', 'distal', False, 0.95, cur_id,
            connections, rng_dip
        )
        dense_points.extend(p_r3)

        rng_tip = add_joint(16, 'ring', weight=1.0, dominant=False)
        connections.append((rc3, rng_tip))
        p_pad_r, cur_id = self._create_fingertip_pad_cluster(pts[16], pts[15], scale, 2, 'ring', cur_id, connections, rng_tip)
        dense_points.extend(p_pad_r)

        # ── 5. PINKY FINGER (14 Points) ──
        pnk_mcp = add_joint(17, 'pinky', weight=0.8, dominant=False)
        p_p1, cur_id, pc1, _, _ = self._interpolate_segment(
            pts[17], pts[18], [0.35, 0.70], [0.50],
            scale * 0.08, 'pinky', 'proximal', False, 0.8, cur_id,
            connections, pnk_mcp
        )
        dense_points.extend(p_p1)

        pnk_pip = add_joint(18, 'pinky', weight=0.85, dominant=False)
        connections.append((pc1, pnk_pip))
        p_p2, cur_id, pc2, _, _ = self._interpolate_segment(
            pts[18], pts[19], [0.35, 0.70], [],
            scale * 0.07, 'pinky', 'intermediate', False, 0.85, cur_id,
            connections, pnk_pip
        )
        dense_points.extend(p_p2)

        pnk_dip = add_joint(19, 'pinky', weight=0.9, dominant=False)
        connections.append((pc2, pnk_dip))
        p_p3, cur_id, pc3, _, _ = self._interpolate_segment(
            pts[19], pts[20], [0.35, 0.70], [],
            scale * 0.06, 'pinky', 'distal', False, 0.9, cur_id,
            connections, pnk_dip
        )
        dense_points.extend(p_p3)

        pnk_tip = add_joint(20, 'pinky', weight=0.95, dominant=False)
        connections.append((pc3, pnk_tip))
        p_pad_p, cur_id = self._create_fingertip_pad_cluster(pts[20], pts[19], scale, 2, 'pinky', cur_id, connections, pnk_tip)
        dense_points.extend(p_pad_p)

        # ── 6. PALM & WEBBING (18 Points) ──
        wrist_id = add_joint(0, 'palm', weight=0.7, dominant=False)

        # Palm center (barycenter of 0, 5, 9, 13, 17)
        palm_ids = (0, 5, 9, 13, 17)
        pc_x = sum(pts[i].x for i in palm_ids) / 5.0
        pc_y = sum(pts[i].y for i in palm_ids) / 5.0
        pc_z = sum(getattr(pts[i], 'z', 0.0) for i in palm_ids) / 5.0
        palm_center_id = cur_id
        dense_points.append(DensePoint(
            id=cur_id, x=pc_x, y=pc_y, z=pc_z,
            region='palm', segment='palm_center',
            is_dominant=False, weight=0.75
        ))
        connections.append((wrist_id, palm_center_id))
        cur_id += 1

        # Metacarpal shafts (4 points)
        for base_joint_id, base_idx in ((idx_mcp, 5), (mid_mcp, 9), (rng_mcp, 13), (pnk_mcp, 17)):
            mx = (wrist.x + pts[base_idx].x) * 0.5
            my = (wrist.y + pts[base_idx].y) * 0.5
            mz = (getattr(wrist, 'z', 0.0) + getattr(pts[base_idx], 'z', 0.0)) * 0.5
            meta_id = cur_id
            dense_points.append(DensePoint(
                id=meta_id, x=mx, y=my, z=mz,
                region='palm', segment='metacarpal',
                is_dominant=False, weight=0.7
            ))
            connections.append((wrist_id, meta_id))
            connections.append((meta_id, base_joint_id))
            connections.append((palm_center_id, base_joint_id))
            cur_id += 1

        # Thenar eminence (2 points)
        t_x1 = wrist.x * 0.4 + pts[1].x * 0.35 + pts[5].x * 0.25
        t_y1 = wrist.y * 0.4 + pts[1].y * 0.35 + pts[5].y * 0.25
        t1_id = cur_id
        dense_points.append(DensePoint(
            id=t1_id, x=t_x1, y=t_y1, z=0.0,
            region='palm', segment='thenar',
            is_dominant=True, weight=0.8
        ))
        connections.append((wrist_id, t1_id))
        connections.append((t1_id, th_cmc))
        cur_id += 1

        t_x2 = pts[1].x * 0.5 + pts[2].x * 0.3 + pts[5].x * 0.2
        t_y2 = pts[1].y * 0.5 + pts[2].y * 0.3 + pts[5].y * 0.2
        t2_id = cur_id
        dense_points.append(DensePoint(
            id=t2_id, x=t_x2, y=t_y2, z=0.0,
            region='palm', segment='thenar',
            is_dominant=True, weight=0.8
        ))
        connections.append((t1_id, t2_id))
        connections.append((t2_id, idx_mcp))
        cur_id += 1

        # Hypothenar eminence (2 points)
        h_x1 = wrist.x * 0.6 + pts[17].x * 0.4
        h_y1 = wrist.y * 0.6 + pts[17].y * 0.4
        h1_id = cur_id
        dense_points.append(DensePoint(
            id=h1_id, x=h_x1, y=h_y1, z=0.0,
            region='palm', segment='hypothenar',
            is_dominant=False, weight=0.7
        ))
        connections.append((wrist_id, h1_id))
        cur_id += 1

        h_x2 = wrist.x * 0.3 + pts[17].x * 0.7
        h_y2 = wrist.y * 0.3 + pts[17].y * 0.7
        h2_id = cur_id
        dense_points.append(DensePoint(
            id=h2_id, x=h_x2, y=h_y2, z=0.0,
            region='palm', segment='hypothenar',
            is_dominant=False, weight=0.7
        ))
        connections.append((h1_id, h2_id))
        connections.append((h2_id, pnk_mcp))
        cur_id += 1

        # Inter-digital webbing (4 points)
        web_pairs = ((th_mcp, 2, idx_mcp, 5), (idx_mcp, 5, mid_mcp, 9),
                     (mid_mcp, 9, rng_mcp, 13), (rng_mcp, 13, pnk_mcp, 17))
        for id_a, i_a, id_b, i_b in web_pairs:
            wx = (pts[i_a].x + pts[i_b].x) * 0.5
            wy = (pts[i_a].y + pts[i_b].y) * 0.5
            wz = (getattr(pts[i_a], 'z', 0.0) + getattr(pts[i_b], 'z', 0.0)) * 0.5
            web_id = cur_id
            dense_points.append(DensePoint(
                id=web_id, x=wx, y=wy, z=wz,
                region='palm', segment='webbing',
                is_dominant=True if i_a == 2 or i_b == 5 else False,
                weight=0.85
            ))
            connections.append((id_a, web_id))
            connections.append((web_id, id_b))
            cur_id += 1

        # Wrist lateral margins (4 points)
        w_dx = pts[9].x - wrist.x
        w_dy = pts[9].y - wrist.y
        w_len = math.hypot(w_dx, w_dy)
        if w_len > 1e-6:
            w_nx, w_ny = -w_dy / w_len, w_dx / w_len
        else:
            w_nx, w_ny = 1.0, 0.0
        w_spread = scale * 0.16
        wrist_margins = []
        for factor in (-1.0, -0.5, 0.5, 1.0):
            wm_id = cur_id
            dense_points.append(DensePoint(
                id=wm_id, x=wrist.x + w_nx * w_spread * factor,
                y=wrist.y + w_ny * w_spread * factor, z=getattr(wrist, 'z', 0.0),
                region='palm', segment='wrist_margin',
                is_dominant=False, weight=0.65
            ))
            wrist_margins.append(wm_id)
            cur_id += 1

        # Connect wrist margins across base
        connections.append((wrist_margins[0], wrist_margins[1]))
        connections.append((wrist_margins[1], wrist_id))
        connections.append((wrist_id, wrist_margins[2]))
        connections.append((wrist_margins[2], wrist_margins[3]))

        self.connections = connections
        return dense_points

    # ─── Frame-to-Frame Optical Flow & Sub-Pixel Refinement ──────────

    def _refine_with_optical_flow(self, current_points: List[DensePoint], frame_rgb: np.ndarray) -> List[DensePoint]:
        """
        Uses sub-pixel Lucas-Kanade optical flow on actual camera pixel intensities
        to measure true physical motion even when camera noise causes neural landmarks to jitter.
        """
        if frame_rgb is None or len(current_points) != 134:
            return current_points

        h, w = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) != 134:
            self.prev_gray = gray
            self.prev_points = current_points
            return current_points

        # Convert dominant finger points to pixel coords for optical flow tracking
        prev_pts_np = np.array([
            [[p.x * w, p.y * h]] for p in self.prev_points if p.is_dominant
        ], dtype=np.float32)

        try:
            next_pts_np, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts_np, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            dom_idx = 0
            refined = []
            for pt in current_points:
                if pt.is_dominant and dom_idx < len(status) and status[dom_idx][0] == 1:
                    flow_x = float(next_pts_np[dom_idx][0][0] / w)
                    flow_y = float(next_pts_np[dom_idx][0][1] / h)

                    # Dynamic blending: release optical flow anchor during active flick
                    prev_p = self.prev_points[pt.id] if (self.prev_points and pt.id < len(self.prev_points)) else None
                    disp_mag = abs(pt.y - prev_p.y) if prev_p else 0.0
                    if disp_mag > self.noise_floor * 1.5:
                        alpha = 0.15  # Active flick: prioritize instant neural landmark response
                    else:
                        alpha = 0.85 if self.is_resting else 0.45  # Resting: anchor firmly against jitter
                    fused_x = alpha * flow_x + (1.0 - alpha) * pt.x
                    fused_y = alpha * flow_y + (1.0 - alpha) * pt.y

                    refined.append(DensePoint(
                        id=pt.id, x=fused_x, y=fused_y, z=pt.z,
                        region=pt.region, segment=pt.segment,
                        is_dominant=pt.is_dominant, weight=pt.weight
                    ))
                    dom_idx += 1
                else:
                    if pt.is_dominant:
                        dom_idx += 1
                    refined.append(pt)

            self.prev_gray = gray
            return refined

        except Exception:
            self.prev_gray = gray
            return current_points

    # ─── Resting vs Micro-Movement Intent Detection ──────────────────

    def process(self, landmarks, frame_rgb=None, now=None) -> Dict[str, Any]:
        """
        Main processing method:
        1. Generates 134 dense points (86 on dominant fingers).
        2. Applies subpixel optical flow refinement when camera frames are available.
        3. Analyzes directional consensus across dominant finger points.
        4. Detects resting stability vs micro-movement intent (UP vs DOWN).
        5. Emits micro-gesture events when intentional finger strokes complete.
        """
        now = time.monotonic() if now is None else now

        if not landmarks or len(landmarks) != 21:
            self.reset()
            return self.get_summary()

        # Step 1: Generate dense anatomical model (134 points)
        dense_pts = self.generate_dense_mesh(landmarks)
        if len(dense_pts) != 134:
            return self.get_summary()

        # Step 2: Optical flow sub-pixel refinement
        if self.enable_optical_flow and frame_rgb is not None:
            dense_pts = self._refine_with_optical_flow(dense_pts, frame_rgb)

        dt = now - self.prev_time if self.prev_time > 0 else 0.033
        dt = max(0.005, min(0.2, dt))

        # Step 3: Dominant finger channels with active lever segmentation
        dominant_pts = [p for p in dense_pts if p.is_dominant]
        index_pts = [p for p in dense_pts if p.region == 'index']
        middle_pts = [p for p in dense_pts if p.region == 'middle']
        thumb_pts = [p for p in dense_pts if p.region == 'thumb']

        # Upper lever segments (intermediate, distal, tip_pad) capture rotation flick consensus without knuckle drag
        active_index = [p for p in index_pts if p.segment in ('intermediate', 'distal', 'tip_pad')]
        active_middle = [p for p in middle_pts if p.segment in ('intermediate', 'distal', 'tip_pad')]
        active_thumb = [p for p in thumb_pts if p.segment in ('intermediate', 'distal', 'tip_pad')]

        channels = {
            'index': (active_index, index_pts),
            'middle': (active_middle, middle_pts),
            'thumb': (active_thumb, thumb_pts),
            'hand': (dominant_pts, dominant_pts),
        }

        # Initialize channel baselines on first view
        for ch_name, (act_pts, _) in channels.items():
            mean_y = float(np.mean([p.y for p in act_pts]))
            if ch_name not in self.channel_baselines:
                self.channel_baselines[ch_name] = mean_y

        micro_gesture = None

        if self.prev_points and len(self.prev_points) == len(dense_pts):
            prev_by_id = {p.id: p for p in self.prev_points}

            # Calculate point-by-point vertical deflections
            dy_index = [curr.y - prev_by_id[curr.id].y for curr in index_pts]
            dy_middle = [curr.y - prev_by_id[curr.id].y for curr in middle_pts]
            dy_thumb = [curr.y - prev_by_id[curr.id].y for curr in thumb_pts]
            dy_dominant = [curr.y - prev_by_id[curr.id].y for curr in dominant_pts]

            self.dominant_deflections['index'] = float(np.mean(dy_index))
            self.dominant_deflections['middle'] = float(np.mean(dy_middle))
            self.dominant_deflections['thumb'] = float(np.mean(dy_thumb))

            # Learn / adapt camera noise floor using median absolute deviations across dominant points
            mean_dom_dy = float(np.mean(dy_dominant))
            point_deviations = [abs(dy - mean_dom_dy) for dy in dy_dominant]
            self.noise_samples.append(float(np.median(point_deviations)))
            if len(self.noise_samples) >= 10:
                self.noise_floor = max(0.0010, min(0.008, float(np.median(self.noise_samples)) * 1.4826))

            # Channel-by-channel motion analysis
            channel_stats = {}
            for ch_name, (act_pts, all_pts) in channels.items():
                act_dys = [curr.y - prev_by_id[curr.id].y for curr in act_pts]
                ch_weights = np.array([p.weight for p in act_pts])
                ch_weights = ch_weights / np.sum(ch_weights)
                ch_dy = float(np.sum(np.array(act_dys) * ch_weights))
                ch_vy = ch_dy / dt

                curr_y = float(np.mean([p.y for p in act_pts]))
                disp_from_base = curr_y - self.channel_baselines[ch_name]

                # Directional consensus among active lever points
                thresh_vote = max(0.0006, self.noise_floor * 0.40)
                n_up = sum(1 for dy in act_dys if dy < -thresh_vote)
                n_down = sum(1 for dy in act_dys if dy > thresh_vote)

                if n_up > n_down:
                    coh = n_up / max(1, len(act_pts))
                    dir_vote = 'UP'
                elif n_down > n_up:
                    coh = n_down / max(1, len(act_pts))
                    dir_vote = 'DOWN'
                else:
                    coh = 0.0
                    dir_vote = None

                channel_stats[ch_name] = {
                    'dy': ch_dy,
                    'vy': ch_vy,
                    'disp': disp_from_base,
                    'coherence': coh,
                    'direction': dir_vote,
                    'pts_count': len(all_pts),
                    'curr_y': curr_y,
                }

            # Select primary channel (Prioritize Index & Middle as primary human scrolling digits)
            active_channel = 'index'
            best_score = channel_stats['index']['coherence'] * abs(channel_stats['index']['vy'])
            for ch_name in ('middle', 'hand', 'thumb'):
                ch_prio = 1.05 if ch_name == 'middle' else (0.85 if ch_name == 'hand' else 0.70)
                score = channel_stats[ch_name]['coherence'] * abs(channel_stats[ch_name]['vy']) * ch_prio
                if score > best_score:
                    best_score = score
                    active_channel = ch_name

            best_stat = channel_stats[active_channel]
            self.micro_velocity = best_stat['vy']
            self.directional_coherence = best_stat['coherence']
            self.velocity_history.append((now, self.micro_velocity, self.directional_coherence))

            # Dynamic thresholds adapted to noise floor, sensitivity, and user's recorded references
            if self.has_references:
                ref_target = self.ref_up_threshold if best_stat['direction'] == 'UP' else self.ref_down_threshold
                min_disp_thresh = max(self.noise_floor * 1.5, min(0.025, (ref_target * 0.75) / self.sensitivity))
            else:
                min_disp_thresh = max(self.noise_floor * 1.5, 0.0035 / self.sensitivity)

            min_vel_thresh = max(self.noise_floor * 8.0, 0.015 / self.sensitivity)

            # Resting vs Active Intent Detection:
            # Stationary hands subject to webcam sensor noise fluctuate within a small band around baseline.
            is_currently_still = (
                abs(best_stat['disp']) < min_disp_thresh * 0.85 or
                abs(best_stat['vy']) < min_vel_thresh * 0.50 or
                best_stat['coherence'] < 0.55
            )

            if is_currently_still:
                # Hand is resting - smoothly adapt baseline to follow natural breathing/drift
                self.resting_stability = min(1.0, self.resting_stability + 0.08)
                self.is_resting = True
                self.micro_intent = None
                self.consecutive_flick_frames = 0
                self.flick_direction = None
                self.micro_stroke_accumulator = 0.0
                if abs(best_stat['disp']) < min_disp_thresh * 0.40 or abs(best_stat['disp']) > min_disp_thresh * 2.0:
                    self.last_fired_direction = None
                # Fast convergence when hand settles in a new wide space position, smooth drift otherwise
                alpha = 0.22 if abs(best_stat['disp']) > min_disp_thresh * 2.2 else 0.06
                for ch_name in self.channel_baselines:
                    self.channel_baselines[ch_name] = (
                        (1.0 - alpha) * self.channel_baselines[ch_name] + alpha * channel_stats[ch_name]['curr_y']
                    )
            else:
                # Active coherent movement detected!
                self.resting_stability = max(0.0, self.resting_stability - 0.15)
                self.is_resting = False

                disp = best_stat['disp']
                direction = best_stat['direction']

                # Return stroke protection: returning from an UP flick toward neutral cannot trigger SWIPE_DOWN
                can_move_up = (self.last_fired_direction != 'DOWN' or disp < -min_disp_thresh * 0.70)
                can_move_down = (self.last_fired_direction != 'UP' or disp > min_disp_thresh * 0.70)
                is_moving_up = (direction == 'UP' and best_stat['vy'] < -min_vel_thresh and can_move_up)
                is_moving_down = (direction == 'DOWN' and best_stat['vy'] > min_vel_thresh and can_move_down)

                if is_moving_up:
                    current_dir = 'UP'
                elif is_moving_down:
                    current_dir = 'DOWN'
                else:
                    current_dir = None

                self.micro_intent = current_dir

                if current_dir is not None:
                    # Evaluate similarity with personal recorded reference positions and whole-movement trajectories
                    ref_sim, ref_step = self._compute_template_match(landmarks, current_dir)
                    traj_sim, traj_step = self._compute_trajectory_match(current_dir, disp, best_stat['vy'])
                    self.last_template_similarity = ref_sim
                    self.last_template_match = ref_step
                    self.last_trajectory_similarity = traj_sim
                    self.last_trajectory_match = traj_step

                    if current_dir == self.flick_direction:
                        self.consecutive_flick_frames += 1
                    else:
                        self.consecutive_flick_frames = 1
                        self.flick_direction = current_dir

                    # Instant trigger criteria:
                    # 1. 2 consecutive frames in same direction (fires in 33-50ms!)
                    # 2. Fast impulse flick on Frame 1 if displacement already decisively exceeds threshold
                    # 3. Whole-movement trajectory match (>= 70% conformance with user's personal recorded strokes)
                    # 4. Reference template match (>= 70% posture similarity)
                    is_instant_impulse = (abs(disp) >= min_disp_thresh * 1.35 and best_stat['coherence'] >= 0.85)
                    is_consecutive_onset = (self.consecutive_flick_frames >= 2 and abs(disp) >= min_disp_thresh * 0.75)
                    is_trajectory_match = (self.has_references and traj_sim >= 0.70 and abs(disp) >= min_disp_thresh * 0.75)
                    is_reference_match = (self.has_references and ref_sim >= 0.70 and abs(disp) >= min_disp_thresh * 0.80)

                    if (is_instant_impulse or is_consecutive_onset or is_trajectory_match or is_reference_match) and (now - self.last_micro_fire_time >= self.micro_cooldown):
                        gesture_type = 'SWIPE_UP' if current_dir == 'UP' else 'SWIPE_DOWN'
                        if self.has_references and traj_sim >= 0.65:
                            source_tag = f"{active_channel}_traj_{current_dir.lower()}"
                        elif self.has_references and ref_sim >= 0.65:
                            source_tag = f"{active_channel}_ref_{current_dir.lower()}"
                        else:
                            source_tag = f"{active_channel}_dense"

                        confidence = min(0.99, max(0.85, 0.80 + best_stat['coherence'] * 0.08 + max(ref_sim, traj_sim) * 0.11))

                        micro_gesture = {
                            'type': gesture_type,
                            'source': source_tag,
                            'confidence': confidence,
                            'velocity': abs(best_stat['vy']),
                            'points_in_consensus': int(best_stat['coherence'] * best_stat['pts_count']),
                            'channel': active_channel,
                            'disp': disp,
                            'reference_step': traj_step or ref_step,
                            'reference_similarity': max(ref_sim, traj_sim),
                            'trajectory_similarity': traj_sim,
                        }
                        self.last_micro_fire_time = now
                        self.last_fired_direction = current_dir
                        self.consecutive_flick_frames = 0
                        self.flick_direction = None
                        self.micro_stroke_accumulator = 0.0

        self.prev_points = dense_pts
        self.points = dense_pts
        self.prev_time = now

        summary = self.get_summary()
        summary['micro_gesture'] = micro_gesture
        return summary

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic and UI state summary."""
        return {
            'total_points': len(self.points),
            'dominant_points': sum(1 for p in self.points if p.is_dominant),
            'finger_points': sum(1 for p in self.points if p.region != 'palm'),
            'is_resting': self.is_resting,
            'resting_stability': self.resting_stability,
            'micro_intent': self.micro_intent,
            'micro_velocity': self.micro_velocity,
            'directional_coherence': self.directional_coherence,
            'noise_floor': self.noise_floor,
            'has_references': self.has_references,
            'ref_up_thresh': self.ref_up_threshold,
            'ref_down_thresh': self.ref_down_threshold,
            'ref_primary_channel': self.ref_primary_channel,
            'template_match': self.last_template_match,
            'template_similarity': self.last_template_similarity,
            'deflections': dict(self.dominant_deflections),
            'dominant_finger_status': {
                'index_deflection': self.dominant_deflections.get('index', 0.0),
                'middle_deflection': self.dominant_deflections.get('middle', 0.0),
                'thumb_deflection': self.dominant_deflections.get('thumb', 0.0),
                'coherence_pct': int(self.directional_coherence * 100),
            }
        }
