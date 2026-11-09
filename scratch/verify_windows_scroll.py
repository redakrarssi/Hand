"""Opt-in desktop smoke: inject only into this script's own local text window."""

import ctypes
import json
from pathlib import Path
import sys
import time
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.gesture_engine import GestureResult
from app.input_simulator import InputSimulator, _get_foreground_handle, _user32


def main():
    root = tk.Tk()
    root.title('HandScroller isolated scroll verification')
    root.geometry('720x520+80+80')
    text = tk.Text(root, wrap='word')
    text.pack(fill='both', expand=True)
    text.insert('1.0', '\n'.join(f'Local scroll verification line {i}' for i in range(500)))
    text.yview_moveto(0.4)
    received = []
    text.bind('<MouseWheel>', lambda event: received.append(event.delta), add='+')
    root.update()
    root.lift()
    root.focus_force()
    text.focus_set()
    root.update()
    hwnd = _user32.GetAncestor(text.winfo_id(), 2)
    if _get_foreground_handle() != hwnd:
        root.destroy()
        raise RuntimeError('Local test window is not foreground; no input was sent')
    _user32.SetCursorPos(text.winfo_rootx() + 300, text.winfo_rooty() + 250)
    root.update()
    sim = InputSimulator('scroll', phone_style=False)
    results = []
    try:
        for gesture, expected_sign in (('SWIPE_DOWN', 1), ('SWIPE_UP', -1)):
            before = text.yview()[0]
            received.clear()
            if _get_foreground_handle() != hwnd:
                raise RuntimeError('Focus changed before injection; aborting')
            action = sim.execute_gesture(GestureResult(gesture))
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and sim._wheel.active:
                root.update()
                sim.update_scroll()
                time.sleep(0.01)
            root.update()
            after = text.yview()[0]
            assert action['success'] and action['status'] == 'completed', action
            assert (after - before) * expected_sign > 0, (before, after)
            assert sum(received) == -expected_sign * 960, received
            results.append(dict(gesture=gesture, before=before, after=after,
                                received_wheel_units=sum(received), events=len(received)))
        # A second owned window exercises real foreground-change cancellation.
        action = sim.execute_gesture(GestureResult('SWIPE_DOWN'))
        other = tk.Toplevel(root)
        other.title('HandScroller focus cancellation verification')
        other.geometry('300x200+850+80')
        tk.Label(other, text='No wheel remainder should reach this window.').pack()
        other_received = []
        other.bind('<MouseWheel>', lambda event: other_received.append(event.delta))
        other.update()
        other.focus_force()
        other.update()
        assert _get_foreground_handle() != hwnd, 'Could not change focus for test'
        assert not sim.update_scroll(sim._wheel.started + 1)
        other.update()
        assert action['status'] == 'interrupted' and action['wheel_units'] == -120, action
        assert not other_received, other_received
        print(json.dumps({'real_window_scroll': results, 'focus_interruption': 'passed'}, indent=2))
    finally:
        sim.cancel_scroll()
        root.destroy()


if __name__ == '__main__':
    main()
