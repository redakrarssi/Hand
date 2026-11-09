"""
Low-Light Image Enhancement Module
Applies adaptive image processing to make hand detection work
reliably in dark/dim environments without needing camera flash.
"""

import cv2
import numpy as np
from app.config import (
    CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE, GAMMA_CORRECTION,
    DENOISE_STRENGTH, ADAPTIVE_BRIGHTNESS
)


class LowLightEnhancer:
    """Enhances camera frames for reliable hand detection in poor lighting."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE
        )
        self.gamma = GAMMA_CORRECTION
        self._build_gamma_table()
        self.brightness_history = []
        self.history_size = 30  # frames for adaptive brightness

    def _build_gamma_table(self):
        """Pre-compute gamma correction lookup table for speed."""
        inv_gamma = 1.0 / self.gamma
        self.gamma_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")

    def measure_brightness(self, frame):
        """Measure average brightness of a frame (0-255)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def get_adaptive_params(self, brightness):
        """Adjust enhancement parameters based on current brightness."""
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > self.history_size:
            self.brightness_history.pop(0)

        avg_brightness = float(np.mean(self.brightness_history))
        has_denoise = DENOISE_STRENGTH > 0

        if avg_brightness < 45:
            # Very dark - aggressive gain boost + strong CLAHE + shadow lift
            return {
                'clahe_clip': 5.5,
                'gamma': 2.4,
                'denoise': 1 if has_denoise else 0,
                'boost': True,
                'gain': min(2.8, max(1.6, 120.0 / max(15.0, avg_brightness))),
                'bias': int(min(32.0, max(10.0, (80.0 - avg_brightness) * 0.45))),
            }
        elif avg_brightness < 85:
            # Dark - moderate boost + CLAHE
            return {
                'clahe_clip': 4.0,
                'gamma': 1.9,
                'denoise': 1 if has_denoise else 0,
                'boost': True,
                'gain': min(2.0, max(1.3, 110.0 / max(25.0, avg_brightness))),
                'bias': int(min(22.0, max(6.0, (85.0 - avg_brightness) * 0.35))),
            }
        elif avg_brightness < 130:
            # Dim - light contrast & midtone enhancement
            return {
                'clahe_clip': 2.8,
                'gamma': 1.4,
                'denoise': 1 if has_denoise else 0,
                'boost': False,
                'gain': 1.15,
                'bias': 4,
            }
        else:
            # Good lighting - minimal processing
            return {
                'clahe_clip': 1.5,
                'gamma': 1.0,
                'denoise': 0,
                'boost': False,
                'gain': 1.0,
                'bias': 0,
            }

    def enhance(self, frame):
        """
        Apply full low-light enhancement pipeline.
        
        Pipeline:
        1. Measure brightness
        2. Adaptive parameter selection
        3. Adaptive luminance gain & bias boost
        4. CLAHE contrast enhancement on L channel
        5. Gamma correction for shadow lifting
        6. Light denoising
        
        Args:
            frame: BGR OpenCV frame
            
        Returns:
            Enhanced BGR frame optimized for hand detection
        """
        if frame is None or frame.size == 0:
            return frame

        brightness = self.measure_brightness(frame)

        if ADAPTIVE_BRIGHTNESS:
            params = self.get_adaptive_params(brightness)
        else:
            params = {
                'clahe_clip': CLAHE_CLIP_LIMIT,
                'gamma': GAMMA_CORRECTION,
                'denoise': DENOISE_STRENGTH,
                'boost': False,
                'gain': 1.0,
                'bias': 0,
            }

        # Step 1: Convert to LAB color space (separate luminance from color)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Step 2: Adaptive luminance gain & bias for poor lighting
        if params.get('boost') or params.get('gain', 1.0) > 1.0:
            gain = params.get('gain', 1.0)
            bias = params.get('bias', 0)
            l_channel = cv2.convertScaleAbs(l_channel, alpha=gain, beta=bias)

        # Step 3: Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(
            clipLimit=params['clahe_clip'],
            tileGridSize=CLAHE_GRID_SIZE
        )
        l_enhanced = clahe.apply(l_channel)

        # Step 4: Merge back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Step 5: Gamma correction (lifts shadows)
        if params['gamma'] != 1.0:
            inv_gamma = 1.0 / params['gamma']
            gamma_table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")
            enhanced = cv2.LUT(enhanced, gamma_table)

        # Step 6: Fast edge-preserving denoising (real-time 60+ FPS)
        if params['denoise'] > 0:
            enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)

        return enhanced

    def enhance_fast(self, frame, params=None):
        """
        Lightweight enhancement for real-time performance.
        Applies adaptive luminance boost + CLAHE + gamma LUT.
        Takes < 1.5ms per frame.
        """
        if frame is None or frame.size == 0:
            return frame

        # Quick CLAHE on luminance path
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Adaptive luminance boost in poor lighting
        if params and (params.get('boost') or params.get('gain', 1.0) > 1.0):
            gain = params.get('gain', 1.0)
            bias = params.get('bias', 0)
            l_channel = cv2.convertScaleAbs(l_channel, alpha=gain, beta=bias)

        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Apply pre-computed gamma table
        if self.gamma != 1.0:
            enhanced = cv2.LUT(enhanced, self.gamma_table)

        return enhanced

    def prepare(self, frame, fast=True):
        """Measure every raw frame; enhance only when the camera is actually dim."""
        brightness = self.measure_brightness(frame[::4, ::4])
        if fast:
            params = self.get_adaptive_params(brightness)
            if brightness >= 130 and not params['boost']:
                return frame
            gamma = params['gamma'] if ADAPTIVE_BRIGHTNESS else GAMMA_CORRECTION
            if gamma != self.gamma:
                self.gamma = gamma
                self._build_gamma_table()
            self.clahe.setClipLimit(params['clahe_clip'])
            return self.enhance_fast(frame, params=params)
        return self.enhance(frame)

    def get_brightness_level(self):
        """Return a human-readable brightness assessment."""
        if not self.brightness_history:
            return "Unknown", 0
        avg = np.mean(self.brightness_history)
        if avg < 40:
            return "Very Dark 🌑", avg
        elif avg < 80:
            return "Dark 🌒", avg
        elif avg < 130:
            return "Dim 🌓", avg
        elif avg < 180:
            return "Good 🌕", avg
        else:
            return "Bright ☀️", avg
