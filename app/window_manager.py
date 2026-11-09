"""
Window and Focus Management for HandScroller.
Ensures browser and target media applications receive gestures reliably,
preventing input from getting trapped in the terminal console or OpenCV HUD.
"""

import ctypes
import logging
import os
import sys
import time

logger = logging.getLogger('HandScroller.WindowManager')

# Win32 Constants
SW_RESTORE = 9
SW_SHOW = 5
VK_MENU = 0x12  # Alt key
KEYEVENTF_KEYUP = 0x0002

_user32 = ctypes.WinDLL('user32', use_last_error=True) if os.name == 'nt' else None
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True) if os.name == 'nt' else None


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


if _user32 is not None:
    _user32.GetForegroundWindow.argtypes = ()
    _user32.GetForegroundWindow.restype = ctypes.c_void_p
    _user32.GetWindowTextLengthW.argtypes = (ctypes.c_void_p,)
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetClassNameW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    _user32.GetClassNameW.restype = ctypes.c_int
    _user32.IsWindowVisible.argtypes = (ctypes.c_void_p,)
    _user32.IsWindowVisible.restype = ctypes.c_bool
    _user32.IsIconic.argtypes = (ctypes.c_void_p,)
    _user32.IsIconic.restype = ctypes.c_bool
    _user32.ShowWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
    _user32.ShowWindow.restype = ctypes.c_bool
    _user32.BringWindowToTop.argtypes = (ctypes.c_void_p,)
    _user32.BringWindowToTop.restype = ctypes.c_bool
    _user32.SetForegroundWindow.argtypes = (ctypes.c_void_p,)
    _user32.SetForegroundWindow.restype = ctypes.c_bool
    _user32.GetWindowThreadProcessId.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    _user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
    _user32.AttachThreadInput.argtypes = (ctypes.c_ulong, ctypes.c_ulong, ctypes.c_bool)
    _user32.AttachThreadInput.restype = ctypes.c_bool
    _user32.keybd_event.argtypes = (ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_size_t)
    _user32.keybd_event.restype = None
    _user32.GetCursorPos.argtypes = (ctypes.POINTER(POINT),)
    _user32.GetCursorPos.restype = ctypes.c_int
    _user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    _user32.SetCursorPos.restype = ctypes.c_int
    _user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(RECT))
    _user32.GetWindowRect.restype = ctypes.c_int
    _user32.mouse_event.argtypes = (ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_size_t)
    _user32.mouse_event.restype = None
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    _user32.EnumWindows.argtypes = (WNDENUMPROC, ctypes.c_void_p)
    _user32.EnumWindows.restype = ctypes.c_bool


CONSOLE_KEYWORDS = [
    'powershell',
    'cmd.exe',
    'command prompt',
    'windows powershell',
    'terminal',
    'bash',
    'mintty',
    'conhost',
    'mingw',
    'run.bat',
]

SYSTEM_EXCLUDE_KEYWORDS = [
    'program manager',
    'default ime',
    'task switching',
    'settings',
    'windows input experience',
    'cortana',
]


def is_console_title(title: str) -> bool:
    """Narrow fallback when a window class cannot be queried."""
    t = (title or '').lower().strip().removeprefix('administrator: ').strip()
    return (t in CONSOLE_KEYWORDS or t in ('windows terminal', 'conhost.exe') or
            t.startswith(('c:\\windows\\system32\\cmd.exe', 'c:\\windows\\system32\\windowspowershell\\')))


def is_console_window(hwnd, title=''):
    if hwnd and _user32 is not None:
        name = ctypes.create_unicode_buffer(256)
        if _user32.GetClassNameW(hwnd, name, len(name)) and name.value:
            return name.value.lower() in ('consolewindowclass', 'cascadia_hosting_window_class',
                                         'mintty', 'virtualconsoleclass')
    return is_console_title(title)


def target_error(hwnd, title):
    """One target policy for dispatch and HUD; never choose another application."""
    if not hwnd or not title:
        return 'No target application is focused'
    if is_hud_title(title):
        return 'HandScroller is focused'
    if is_console_window(hwnd, title):
        return 'Terminal is focused'
    return None


def is_hud_title(title: str) -> bool:
    """Check if title matches the HandScroller preview/HUD window."""
    if not title:
        return False
    # OpenCV's Win32 backend can decode the UTF-8 emoji as legacy code-page
    # characters. Ignore that non-ASCII prefix, not arbitrary document suffixes.
    t = title.lower().strip()
    while t and (not t[0].isascii() or t[0].isspace()):
        t = t[1:]
    return t in ('handscroller', 'handscroller (privacy mode)')


def get_window_title(hwnd) -> str:
    """Get window title by HWND."""
    if not hwnd or _user32 is None:
        return ""
    try:
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def get_foreground_title() -> str:
    """Get title of current foreground window."""
    if _user32 is None:
        return ""
    try:
        hwnd = _user32.GetForegroundWindow()
        return get_window_title(hwnd)
    except Exception:
        return ""


def ensure_cursor_in_window(hwnd):
    """Intentionally does nothing: HandScroller never moves the user's cursor."""
    return


def activate_window(hwnd) -> bool:
    """
    Forcefully and reliably brings a window to the foreground on Windows.
    Bypasses UIPI foreground locking using thread attachment and simulated Alt key event.
    """
    if not hwnd or _user32 is None:
        return False

    try:
        current_fg = _user32.GetForegroundWindow()
        if current_fg == hwnd:
            ensure_cursor_in_window(hwnd)
            return True

        # Restore window if minimized
        if _user32.IsIconic(hwnd):
            _user32.ShowWindow(hwnd, SW_RESTORE)
        else:
            _user32.ShowWindow(hwnd, SW_SHOW)

        # Thread attachment bypass for Windows SetForegroundWindow lock
        fg_thread = _user32.GetWindowThreadProcessId(current_fg, None) if current_fg else 0
        target_thread = _user32.GetWindowThreadProcessId(hwnd, None)
        current_thread = _kernel32.GetCurrentThreadId() if _kernel32 else 0

        attached_fg = False
        attached_target = False

        if fg_thread and current_thread and fg_thread != current_thread:
            attached_fg = _user32.AttachThreadInput(current_thread, fg_thread, True)
        if target_thread and current_thread and target_thread != current_thread:
            attached_target = _user32.AttachThreadInput(current_thread, target_thread, True)

        # Alt key trick: Windows allows foreground change if menu key is pressed
        _user32.keybd_event(VK_MENU, 0, 0, 0)
        _user32.BringWindowToTop(hwnd)
        success = _user32.SetForegroundWindow(hwnd)
        _user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

        # Detach threads
        if attached_fg:
            _user32.AttachThreadInput(current_thread, fg_thread, False)
        if attached_target:
            _user32.AttachThreadInput(current_thread, target_thread, False)

        ensure_cursor_in_window(hwnd)
        return bool(success)
    except Exception as e:
        logger.debug("Failed to activate window %s: %s", hwnd, e)
        return False


def find_target_browser_hwnd():
    """
    Enumerate all visible windows and return the best browser or media application window.
    Prioritizes:
    1. HandScroller Interactive Test Playground
    2. TikTok / YouTube Shorts / Reels
    3. Chrome / Edge / Firefox / Brave / Opera
    4. Other non-terminal, non-HUD windows
    """
    if _user32 is None:
        return None

    candidates = []

    def enum_cb(hwnd, lparam):
        try:
            if not _user32.IsWindowVisible(hwnd):
                return True
            title = get_window_title(hwnd)
            if not title or len(title.strip()) < 2:
                return True

            t_lower = title.lower()

            # Exclude HUD and consoles and system items
            if is_hud_title(t_lower) or is_console_title(t_lower):
                return True
            if any(sys_kw in t_lower for sys_kw in SYSTEM_EXCLUDE_KEYWORDS):
                return True

            # Score candidates
            score = 0
            if 'handscroller interactive' in t_lower or 'scroll test' in t_lower:
                score = 100
            elif any(k in t_lower for k in ('tiktok', 'shorts', 'youtube', 'reels', 'instagram')):
                score = 80
            elif any(b in t_lower for b in ('chrome', 'edge', 'firefox', 'brave', 'opera', 'vivaldi')):
                score = 60
            else:
                score = 10

            candidates.append((score, hwnd, title))
        except Exception:
            pass
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    _user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

    if not candidates:
        return None

    # Sort descending by score
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_hwnd, best_title = candidates[0]
    logger.debug("Found target window candidate: %s (score=%d)", best_title, best_score)
    return best_hwnd


def activate_target_browser() -> bool:
    """Find and focus the user's browser or active target window."""
    hwnd = find_target_browser_hwnd()
    if hwnd:
        title = get_window_title(hwnd)
        activated = activate_window(hwnd)
        if activated:
            logger.info("🎯 Focused target window: '%s'", title)
            return True
    return False


def ensure_target_active() -> bool:
    """
    Check if a valid target is active. If focus is on the terminal or HUD,
    automatically switch focus to the browser window.
    """
    current_title = get_foreground_title()
    if not current_title or is_console_title(current_title) or is_hud_title(current_title):
        logger.debug("Foreground is '%s' (not a target app), finding browser...", current_title)
        return activate_target_browser()
    return True
