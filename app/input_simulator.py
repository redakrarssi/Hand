"""
Input Simulator Module
Simulates keyboard and mouse actions to control scrolling
across different platforms (TikTok, YouTube Shorts, Instagram, etc.)
"""

import ctypes
import os
import time
import logging
from app.config import (
    SCROLL_MODE, SCROLL_AMOUNT, KEYBOARD_KEY_DOWN,
    KEYBOARD_KEY_UP, KEYBOARD_KEY_PAUSE, PLATFORM_KEYWORDS,
    SOUND_ENABLED, PHONE_STYLE_SCROLL, WINDOW_NAME
)

logger = logging.getLogger(__name__)


# ─── Windows API Constants ───────────────────────────────────────
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_WHEEL = 0x0800
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001
WHEEL_DELTA = 120

# Virtual key codes
VK_CODES = {
    'up': 0x26,
    'down': 0x28,
    'left': 0x25,
    'right': 0x27,
    'space': 0x20,
    'enter': 0x0D,
    'pageup': 0x21,
    'pagedown': 0x22,
    'escape': 0x1B,
    'tab': 0x09,
    'j': 0x4A,    # YouTube next
    'k': 0x4B,    # YouTube pause
    'l': 0x4C,    # YouTube forward
    'f': 0x46,    # Fullscreen
}


# ─── Windows Input Structures ────────────────────────────────────
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", INPUT_UNION),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


_user32 = ctypes.WinDLL('user32', use_last_error=True) if os.name == 'nt' else None
if _user32 is not None:
    _user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
    _user32.SendInput.restype = ctypes.c_uint
    _user32.GetForegroundWindow.argtypes = ()
    _user32.GetForegroundWindow.restype = ctypes.c_void_p
    _user32.GetWindowTextLengthW.argtypes = (ctypes.c_void_p,)
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetCursorPos.argtypes = (ctypes.POINTER(POINT),)
    _user32.GetCursorPos.restype = ctypes.c_int
    _user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    _user32.SetCursorPos.restype = ctypes.c_int
    _user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(RECT))
    _user32.GetWindowRect.restype = ctypes.c_int
    _user32.WindowFromPoint.argtypes = (POINT,)
    _user32.WindowFromPoint.restype = ctypes.c_void_p
    _user32.GetAncestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    _user32.GetAncestor.restype = ctypes.c_void_p
    _user32.mouse_event.argtypes = (ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_size_t)
    _user32.mouse_event.restype = None
    _user32.keybd_event.argtypes = (ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_size_t)
    _user32.keybd_event.restype = None
    _user32.MapVirtualKeyW.argtypes = (ctypes.c_uint, ctypes.c_uint)
    _user32.MapVirtualKeyW.restype = ctypes.c_uint


def _ensure_cursor_over_target(hwnd=None):
    """True only if the cursor is already over the focused app; never moves it."""
    if _user32 is None:
        return False
    try:
        hwnd = hwnd or _user32.GetForegroundWindow()
        if not hwnd or _user32.GetForegroundWindow() != hwnd:
            return False
        rect = RECT()
        pt = POINT()
        if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)) or not _user32.GetCursorPos(ctypes.byref(pt)):
            return False
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w <= 80 or h <= 160:
            return False

        def over_target(point):
            hovered = _user32.WindowFromPoint(point)
            return bool(hovered and _user32.GetAncestor(hovered, 2) == hwnd)

        # Never move the user's cursor: wheel input only goes out while the
        # pointer is already over the focused window.
        return _user32.GetForegroundWindow() == hwnd and over_target(pt)
    except Exception as exc:
        logger.debug('Cannot target wheel input: %s', exc)
        return False


def _send_input(input_struct):
    """Check Windows acceptance; this does not prove the target page moved."""
    if _user32 is None:
        return False
    try:
        ctypes.set_last_error(0)
        if _user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(INPUT)) == 1:
            return True
    except Exception as exc:
        logger.debug('SendInput dispatch error: %s', exc)
    # A void legacy API cannot verify delivery, and retrying can duplicate input.
    logger.warning('Windows did not accept input (error=%s). Focus a normal browser/app window.',
                   ctypes.get_last_error())
    return False


def _get_foreground_window_title():
    """Get the title of the currently active window."""
    try:
        if _user32 is None:
            return ''
        hwnd = _user32.GetForegroundWindow()
        length = _user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.lower()
    except Exception:
        return ""


def _is_hud_title(title):
    # A project or browser page named HandScroller is still a valid target.
    t = title.lower().lstrip('✋ ').strip()
    return t == WINDOW_NAME.lower().lstrip('✋ ').strip() or t == 'handscroller (privacy mode)'


def _get_foreground_handle():
    return _user32.GetForegroundWindow() if _user32 is not None else None


class InputSimulator:
    """
    Simulates scroll and keyboard input for hands-free control.
    
    Supports three modes:
    - "scroll": Mouse wheel simulation (universal)
    - "keyboard": Arrow key simulation (best for video platforms)
    - "smart": Auto-detects platform from window title
    """

    def __init__(self, mode=None, phone_style=None):
        self.mode = mode or SCROLL_MODE
        self.scroll_amount = SCROLL_AMOUNT
        self.phone_style = PHONE_STYLE_SCROLL if phone_style is None else phone_style
        self.action_log = []
        self.total_actions = 0
        self.last_action_time = 0
        self._paused = False
        from app.smooth_scroll import SmoothWheel
        self._wheel = SmoothWheel(self._send_wheel_delta, _get_foreground_handle)
        self._scroll_action = None
        self._scroll_error = None
        self._input_target = None

    def _detect_platform_mode(self):
        """Detect the best input mode based on active window."""
        title = _get_foreground_window_title().lower()
        for keyword, mode in PLATFORM_KEYWORDS.items():
            if keyword in title:
                return mode
        return "scroll"  # Default to mouse scroll

    def _get_effective_mode(self):
        """Get the current effective input mode."""
        if self.mode == "smart":
            return self._detect_platform_mode()
        return self.mode

    def _simulate_key_press(self, key_name):
        """Simulate a key press and release."""
        vk_code = VK_CODES.get(key_name.lower())
        if vk_code is None:
            logger.warning(f"Unknown key: {key_name}")
            return False

        scan_code = _user32.MapVirtualKeyW(vk_code, 0) if _user32 is not None else 0
        extended = KEYEVENTF_EXTENDEDKEY if key_name in ('up', 'down', 'left', 'right', 'pageup', 'pagedown') else 0

        # Key down
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk_code
        inp.union.ki.wScan = scan_code
        inp.union.ki.dwFlags = extended
        if self._input_target is not None and _get_foreground_handle() != self._input_target:
            return False
        down_ok = _send_input(inp)

        # Key up
        inp2 = INPUT()
        inp2.type = INPUT_KEYBOARD
        inp2.union.ki.wVk = vk_code
        inp2.union.ki.wScan = scan_code
        inp2.union.ki.dwFlags = extended | KEYEVENTF_KEYUP
        up_ok = _send_input(inp2)

        return down_ok and up_ok

    def _send_wheel_delta(self, amount):
        if _ensure_cursor_over_target(self._input_target) is False:
            return False
        if self._input_target is not None and _get_foreground_handle() != self._input_target:
            return False
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dwFlags = MOUSEEVENTF_WHEEL
        inp.union.mi.mouseData = ctypes.c_ulong(amount & 0xFFFFFFFF)
        return _send_input(inp)

    def _simulate_scroll(self, direction):
        """Send the first wheel step now; ease the remainder on subsequent frames."""
        self.cancel_scroll()
        self._scroll_action = None
        if _ensure_cursor_over_target(self._input_target) is False:
            # The wheel only reaches the window under the pointer and we never
            # move the pointer, so page with the arrow keys instead of failing.
            self._wheel.sent = 0
            return self._simulate_key_press(KEYBOARD_KEY_UP if direction > 0 else KEYBOARD_KEY_DOWN)
        return self._wheel.start(direction * self.scroll_amount * WHEEL_DELTA)

    def _interrupt_scroll(self, error):
        action = self._scroll_action
        if action is not None and action.get('status') == 'scrolling':
            action.update(success=False, status='interrupted', wheel_units=self._wheel.sent,
                          error=error, description='Scroll interrupted - focus your browser/page')
            self.total_actions -= 1
            self._scroll_error = action

    def update_scroll(self, now=None):
        ok = self._wheel.update(now)
        if self._scroll_action is not None:
            self._scroll_action['wheel_units'] = self._wheel.sent
            if not ok:
                self._interrupt_scroll(self._wheel.error)
            elif not self._wheel.active and self._scroll_action.get('status') == 'scrolling':
                self._scroll_action['status'] = 'completed'
        return ok

    def pop_scroll_error(self):
        action, self._scroll_error = self._scroll_error, None
        return action

    def cancel_scroll(self):
        if self._wheel.active:
            self._interrupt_scroll('Scrolling cancelled before completion')
        self._wheel.cancel()

    def _play_feedback_sound(self):
        """Play a subtle feedback sound on gesture detection."""
        if SOUND_ENABLED:
            try:
                import winsound
                winsound.Beep(800, 50)  # Short, subtle beep
            except Exception:
                pass

    def undo_action(self, action_info):
        """
        Reverse/undo the previous action on the target application.
        If we scrolled down, scrolls up by the exact same amount.
        If we pressed Down Arrow, presses Up Arrow.
        """
        if not action_info or not (action_info.get('success') or action_info.get('wheel_units')):
            return False
        try:
            if action_info.get('target_window') is not None and action_info['target_window'] != _get_foreground_handle():
                self.cancel_scroll()
                return False
            self._input_target = _get_foreground_handle()
            if not self._input_target:
                return False
            if 'wheel_units' in action_info:
                self.cancel_scroll()
                # Undo exactly the units actually sent, including a partial animation.
                return bool(action_info['wheel_units']) and self._send_wheel_delta(-action_info['wheel_units'])
            scroll_dir = action_info.get('scroll')
            key = action_info.get('key')
            if scroll_dir == 'down':
                return self._simulate_scroll(1)
            elif scroll_dir == 'up':
                return self._simulate_scroll(-1)
            elif key == KEYBOARD_KEY_DOWN:
                return self._simulate_key_press(KEYBOARD_KEY_UP)
            elif key == KEYBOARD_KEY_UP:
                return self._simulate_key_press(KEYBOARD_KEY_DOWN)
            elif key == 'left':
                return self._simulate_key_press('right')
            elif key == 'right':
                return self._simulate_key_press('left')
            elif action_info.get('gesture') == 'PALM_HOLD':
                return self._simulate_key_press(KEYBOARD_KEY_PAUSE)
            return False
        except Exception as e:
            logger.warning("Failed to undo action: %s", e)
            return False

    def execute_gesture(self, gesture):
        """
        Execute the appropriate action for a detected gesture.
        
        Args:
            gesture: GestureResult from the gesture engine
            
        Returns:
            dict with action details
        """
        if self._paused:
            return {'action': 'paused', 'success': False}

        now = time.time()
        mode = self._get_effective_mode()
        action_info = {
            'gesture': gesture.type,
            'mode': mode,
            'success': False,
            'timestamp': now,
        }

        title = _get_foreground_window_title().lower()
        from app.window_manager import target_error
        target = _get_foreground_handle()
        error = target_error(target, title)
        if error:
            action_info.update(description='Click your browser/page to scroll', error=error, status='failed')
            self.action_log.append(action_info)
            self.action_log = self.action_log[-50:]
            return action_info

        self.cancel_scroll()
        self._input_target = target
        action_info['target_window'] = target
        try:
            source = getattr(gesture, 'source', None)
            if source == 'touch_thumb_index' or gesture.type == 'TOUCH_DOWN':
                # Big finger (Thumb) touching Index finger -> Always Scroll DOWN
                if mode == "keyboard":
                    success = self._simulate_key_press(KEYBOARD_KEY_DOWN)
                    action_info['key'] = KEYBOARD_KEY_DOWN
                else:
                    success = self._simulate_scroll(-1)
                    action_info['scroll'] = 'down'
                action_info['description'] = "Scroll DOWN (Thumb + Index Touch)"
                action_info['success'] = success

            elif source == 'touch_thumb_middle' or gesture.type == 'TOUCH_UP':
                # Big finger (Thumb) touching Middle finger -> Always Scroll UP
                if mode == "keyboard":
                    success = self._simulate_key_press(KEYBOARD_KEY_UP)
                    action_info['key'] = KEYBOARD_KEY_UP
                else:
                    success = self._simulate_scroll(1)
                    action_info['scroll'] = 'up'
                action_info['description'] = "Scroll UP (Thumb + Middle Touch)"
                action_info['success'] = success

            elif gesture.type == "SWIPE_UP":
                # In Phone Style: Flicking from down to up pulls next video into view (Scroll Down / Down Arrow)
                if self.phone_style:
                    if mode == "keyboard":
                        success = self._simulate_key_press(KEYBOARD_KEY_DOWN)
                        action_info['key'] = KEYBOARD_KEY_DOWN
                    else:
                        success = self._simulate_scroll(-1)
                        action_info['scroll'] = 'down'
                    action_info['description'] = "Scroll DOWN / Next (hand UP)"
                else:
                    if mode == "keyboard":
                        success = self._simulate_key_press(KEYBOARD_KEY_UP)
                        action_info['key'] = KEYBOARD_KEY_UP
                    else:
                        success = self._simulate_scroll(1)
                        action_info['scroll'] = 'up'
                    action_info['description'] = "Scroll UP (hand UP)"
                action_info['success'] = success

            elif gesture.type == "SWIPE_DOWN":
                # In Phone Style: Flicking from up to down pulls previous video into view (Scroll Up / Up Arrow)
                if self.phone_style:
                    if mode == "keyboard":
                        success = self._simulate_key_press(KEYBOARD_KEY_UP)
                        action_info['key'] = KEYBOARD_KEY_UP
                    else:
                        success = self._simulate_scroll(1)
                        action_info['scroll'] = 'up'
                    action_info['description'] = "Scroll UP / Previous (hand DOWN)"
                else:
                    if mode == "keyboard":
                        success = self._simulate_key_press(KEYBOARD_KEY_DOWN)
                        action_info['key'] = KEYBOARD_KEY_DOWN
                    else:
                        success = self._simulate_scroll(-1)
                        action_info['scroll'] = 'down'
                    action_info['description'] = "Scroll DOWN (hand DOWN)"
                action_info['success'] = success

            elif gesture.type == "SWIPE_LEFT":
                if mode == "keyboard":
                    success = self._simulate_key_press('left')
                else:
                    success = self._simulate_scroll(-1)
                action_info['success'] = success
                action_info['description'] = "⬅️ Swipe Left"

            elif gesture.type == "SWIPE_RIGHT":
                if mode == "keyboard":
                    success = self._simulate_key_press('right')
                else:
                    success = self._simulate_scroll(1)
                action_info['success'] = success
                action_info['description'] = "➡️ Swipe Right"

            elif gesture.type == "PALM_HOLD":
                success = self._simulate_key_press(KEYBOARD_KEY_PAUSE)
                action_info['success'] = success
                action_info['description'] = "⏸️ Pause/Play"

            if action_info['success']:
                self.total_actions += 1
                self.last_action_time = now
                self._play_feedback_sound()
            else:
                action_info['description'] = 'Input failed - focus your browser/page'
                action_info['error'] = 'Windows did not accept the requested input'

        except Exception as e:
            logger.error(f"Input simulation error: {e}")
            action_info['error'] = str(e)
            action_info['description'] = 'Input failed - check the application log'

        action_info['status'] = 'completed' if action_info['success'] else 'failed'
        if action_info['success'] and (action_info.get('scroll') or
                                      (mode == 'scroll' and gesture.type in ('SWIPE_LEFT', 'SWIPE_RIGHT'))):
            action_info['wheel_units'] = self._wheel.sent
            action_info['status'] = 'scrolling' if self._wheel.active else 'completed'
            self._scroll_action = action_info
        self.action_log.append(action_info)
        if len(self.action_log) > 50:
            self.action_log.pop(0)

        return action_info

    def set_mode(self, mode):
        """Change input mode."""
        if mode in ("scroll", "keyboard", "smart"):
            self.cancel_scroll()
            self.mode = mode
            logger.info(f"Input mode changed to: {mode}")

    def set_scroll_amount(self, amount):
        """Adjust scroll amount (1-20)."""
        self.scroll_amount = max(1, min(20, amount))

    def pause(self):
        """Pause input simulation."""
        self._paused = True
        self.cancel_scroll()

    def resume(self):
        """Resume input simulation."""
        self._paused = False

    def is_paused(self):
        """Check if simulation is paused."""
        return self._paused

    def get_stats(self):
        """Get simulation statistics."""
        from app.window_manager import target_error
        title = _get_foreground_window_title().lower()
        return {
            'mode': self.mode,
            'effective_mode': self._get_effective_mode(),
            'total_actions': self.total_actions,
            'scroll_amount': self.scroll_amount,
            'paused': self._paused,
            'last_action': self.action_log[-1] if self.action_log else None,
            'phone_style': self.phone_style,
            'target_ready': target_error(_get_foreground_handle(), title) is None,
        }

    def get_active_window(self):
        """Get info about the currently active window."""
        title = _get_foreground_window_title()
        mode = self._detect_platform_mode()
        platform = "Unknown"
        for keyword in PLATFORM_KEYWORDS:
            if keyword in title:
                platform = keyword.capitalize()
                break
        return {
            'title': title[:60],
            'platform': platform,
            'mode': mode,
        }
