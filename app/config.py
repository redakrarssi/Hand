"""
HandScroller Configuration
All tunable parameters for gesture detection, camera, and actions.
"""

# ─── Camera Settings ─────────────────────────────────────────────
CAMERA_INDEX = 0                    # Default camera (0 = built-in webcam)
CAMERA_WIDTH = 640                  # Capture width (lower = faster)
CAMERA_HEIGHT = 480                 # Capture height
CAMERA_FPS = 30                     # Target FPS
CAMERA_EXPOSURE = -6                # Used only when manual exposure is selected
CAMERA_AUTO_EXPOSURE = 1.0          # 1 = automatic, 0 = manual; translated per Windows backend
CAMERA_BACKEND = "AUTO"             # "AUTO" uses MSMF for 30 FPS on Windows, falling back to DSHOW

# ─── Low-Light Enhancement ───────────────────────────────────────
LOW_LIGHT_ENABLED = True            # Enable adaptive low-light processing
CLAHE_CLIP_LIMIT = 3.5              # Contrast limit for CLAHE (higher = more contrast)
CLAHE_GRID_SIZE = (8, 8)            # Tile grid for CLAHE
GAMMA_CORRECTION = 1.6              # Gamma value (>1 brightens dark areas)
DENOISE_STRENGTH = 0                # Skip costly denoising for lower latency
ADAPTIVE_BRIGHTNESS = True          # Auto-adjust enhancement based on frame brightness

# ─── MediaPipe Hand Detection ────────────────────────────────────
MAX_HANDS = 1                       # Track 1 hand
MIN_DETECTION_CONFIDENCE = 0.16     # High sensitivity for wide space, distant hands, and dim lighting
MIN_TRACKING_CONFIDENCE = 0.16      # Stable tracking across entire camera view and periphery

# ─── Gesture Detection ───────────────────────────────────
MOTION_GESTURES_ENABLED = False     # Default Touch-Safety mode (Thumb+Index=DOWN, Thumb+Middle=UP); press G or use --motion to enable hand swipes
PHONE_STYLE_SCROLL = False          # Natural: Hand UP = Scroll UP / Previous, Hand DOWN = Scroll DOWN / Next
GESTURE_COOLDOWN = 0.26             # Fast, fluid cooldown between strokes
POSITION_HISTORY_SIZE = 15
PALM_HOLD_ENABLED = False           # Relaxing an open hand should not pause content
PALM_HOLD_TIME = 1.5                # Seconds to hold still if enabled
PALM_HOLD_STILL_THRESHOLD = 0.03    # Max movement to count as "still" for palm hold

# Pure deterministic intent detection across the full view (no virtual boxes).
# Immediate 1ms detection upon appearing in camera view.
INTENT_CALIBRATION_TIME = 0.12
INTENT_SETTLE_TIME = 0.06
INTENT_MIN_STROKE_TIME = 0.05
INTENT_MAX_STROKE_TIME = 2.5
INTENT_FINGER_DISTANCE = 0.038      # Relaxed finger flick distance (~12-18px)
INTENT_HAND_DISTANCE = 0.055        # Relaxed whole-hand movement distance
INTENT_DIRECTION_RATIO = 0.75       # Allows natural slightly-diagonal human finger flicks
INTENT_TRACKING_GAP = 0.35          # Tolerate brief tracking interruptions; reset on longer gap
INTENT_RECOVERY_TIMEOUT = 0.90      # Fast re-arm to adopt a new relaxed resting posture

# ─── Scroll Actions ─────────────────────────────────────────────
SCROLL_MODE = "smart"               # "smart" auto-detects: keyboard for TikTok/Shorts, wheel for web & test page
SCROLL_AMOUNT = 5                   # Wheel steps per gesture; gentle enough to page one reel at a time
SMOOTH_SCROLL_DURATION = 0.16       # Smooth wheel easing without blocking tracking
KEYBOARD_KEY_DOWN = "down"          # Key for next video / scroll down
KEYBOARD_KEY_UP = "up"              # Key for previous video / scroll up
KEYBOARD_KEY_PAUSE = "space"        # Key for pause/play

# Smart mode platform detection keywords in window titles
PLATFORM_KEYWORDS = {
    "tiktok": "keyboard",           # TikTok works best with arrow keys
    "shorts": "keyboard",           # Avoid arrow keys on regular YouTube videos
    "reels": "keyboard",            # Instagram Reels works best with arrow keys
    "instagram": "keyboard",        # Instagram Reels pages cleanly with arrow keys
    "facebook": "scroll",           # Facebook Reels uses scroll
    "twitter": "scroll",            # X/Twitter uses scroll
    "reddit": "scroll",             # Reddit uses scroll
}

# ─── UI / Overlay ────────────────────────────────────────────────
SHOW_CAMERA_PREVIEW = True          # Show HUD window with gesture indicators
SHOW_CAMERA_FEED = False            # Privacy Mode: Camera video feed is completely hidden (press V to preview)
PRIVACY_MODE = True                 # True = Black privacy canvas with hand landmarks; False = Live webcam video
WINDOW_WIDTH = 520                  # Readable gesture status and framing hints
WINDOW_HEIGHT = 390
MIRROR_MODE = True                  # Mirror hand motion (natural feel)
OVERLAY_OPACITY = 0.7               # Overlay transparency
SHOW_LANDMARKS = True               # Draw hand landmarks / skeleton
SHOW_FPS = True                     # Show FPS counter
SHOW_GESTURE_FEEDBACK = True        # Show gesture arrows/text
WINDOW_NAME = "HandScroller (Privacy Mode)"  # ASCII title avoids OpenCV Win32 encoding corruption
MINIMAP_ENABLED = True              # Show small hand position indicator

# ─── Colors (BGR format for OpenCV) ──────────────────────────────
COLOR_PRIMARY = (255, 140, 50)      # Bright blue-ish
COLOR_SECONDARY = (50, 220, 160)    # Teal green
COLOR_ACCENT = (80, 80, 255)        # Red accent
COLOR_SUCCESS = (80, 220, 80)       # Green
COLOR_WARNING = (50, 180, 255)      # Orange
COLOR_TEXT = (255, 255, 255)        # White
COLOR_BG = (30, 30, 30)             # Dark background
COLOR_LANDMARK = (255, 180, 50)     # Landmark dots
COLOR_CONNECTION = (100, 255, 100)  # Landmark connections

# ─── Sound Feedback ──────────────────────────────────────────────
SOUND_ENABLED = False               # Beep on gesture (can be annoying)
