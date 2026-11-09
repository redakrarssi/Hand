"""Time-based wheel easing driven by the main loop, without sleeps or workers."""

import time

from app.config import SMOOTH_SCROLL_DURATION


class SmoothWheel:
    def __init__(self, send_delta, get_target, duration=SMOOTH_SCROLL_DURATION):
        self.send_delta = send_delta
        self.get_target = get_target
        self.duration = duration
        self.active = False
        self.sent = 0
        self.error = None

    def cancel(self):
        self.active = False

    def start(self, amount, now=None):
        self.cancel()
        self.total = int(amount)
        self.sent = 0
        self.error = None
        self.target = self.get_target()
        self.started = time.monotonic() if now is None else now
        self.first = min(120, abs(self.total)) * (1 if self.total > 0 else -1)
        if not self.send_delta(self.first):
            self.error = 'Windows did not accept wheel input'
            return False
        self.sent = self.first
        self.active = self.sent != self.total
        return True

    def update(self, now=None):
        if not self.active:
            return True
        if self.get_target() != self.target:
            self.error = 'Scrolling stopped because the focused window changed'
            self.cancel()
            return False
        now = time.monotonic() if now is None else now
        progress = min(1.0, max(0.0, (now - self.started) / self.duration))
        desired = self.first + round((self.total - self.first) * (1 - (1 - progress) ** 3))
        delta = desired - self.sent
        if delta and not self.send_delta(delta):
            self.error = 'Windows did not accept wheel input'
            self.cancel()
            return False
        self.sent = desired
        if progress >= 1:
            self.cancel()
        return True
