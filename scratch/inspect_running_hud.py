"""Capture only HandScroller's privacy HUD and exercise its recalibrate control."""

import ctypes
from pathlib import Path
import time
from PIL import ImageGrab

user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
user32.FindWindowW.restype = ctypes.c_void_p
user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
user32.PostMessageW.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t)

class RECT(ctypes.Structure):
    _fields_ = [(name, ctypes.c_long) for name in ('left', 'top', 'right', 'bottom')]

end = time.monotonic() + 25
hwnd = None
while time.monotonic() < end:
    hwnd = user32.FindWindowW(None, 'HandScroller (Privacy Mode)')
    if hwnd:
        break
    time.sleep(0.25)
if not hwnd:
    raise RuntimeError('HandScroller HUD did not appear')
# Target only our HUD, rather than sending a global key to the desktop.
user32.PostMessageW(hwnd, 0x102, ord('c'), 0)
time.sleep(1)
rect = RECT()
if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
    raise RuntimeError('Could not get HUD rectangle')
path = Path(__file__).with_name('fixed_handscroller_hud.png')
ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom)).save(path)
print(f'HUD hwnd={hwnd}; recalibrate key posted; screenshot={path}')
