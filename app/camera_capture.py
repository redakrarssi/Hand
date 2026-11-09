"""Keep only the newest camera frame, so slow inference never builds a queue."""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class LatestFrameCamera:
    def __init__(self, capture):
        self.capture = capture
        self.timestamp = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._timestamp = None
        self._sequence = self._consumed = 0
        self._finished = False
        self._thread = threading.Thread(target=self._capture, name='HandScrollerCamera', daemon=True)
        self._thread.start()

    def _capture(self):
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    failures += 1
                    if failures >= 15:
                        break
                    self._stop.wait(0.02)
                    continue
                failures = 0
                with self._condition:
                    self._frame = frame
                    self._timestamp = time.monotonic()
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception:
            logger.exception('Camera capture failed')
        finally:
            self.capture.release()
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def read(self, timeout=0.045):
        """True means a fresh frame, None a short timeout, False end of capture."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > self._consumed or self._finished, timeout)
            if self._sequence > self._consumed:
                self._consumed = self._sequence
                self.timestamp = self._timestamp
                return True, self._frame
            return (False if self._finished else None), None

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        # Only the capture thread releases the device, avoiding read/release races.
