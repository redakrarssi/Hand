"""Landmark-only regressions for quick repeats; never inject desktop input."""

import math
import unittest

from app.config import GESTURE_COOLDOWN
from test_relaxed_scrolling import Replay, make_hand


class RepeatRecoveryTests(unittest.TestCase):
    def replay(self, fps, threshold):
        replay = Replay(fps)
        if threshold is not None:
            replay.tracker.set_learned_thresholds(threshold, threshold)
        replay.fire_times = []
        update = replay.tracker.update

        def timed_update(landmarks, now):
            event = update(landmarks, now)
            if event:
                replay.fire_times.append(now)
            return event

        replay.tracker.update = timed_update
        return replay.feed(0.5)

    def stroke(self, replay, sign, source, outward=0.16, back=0.12):
        if source == 'hand':
            replay.feed(outward, hand_y=lambda t: sign * 0.035 * t)
            replay.feed(back, hand_y=lambda t: sign * 0.035 * (1 - t))
        else:
            replay.feed(outward, finger=source, dy=lambda t: sign * 0.035 * t)
            replay.feed(back, finger=source, dy=lambda t: sign * 0.035 * (1 - t))
        replay.feed(0.03)

    def check_sequence(self, signs):
        for fps in (15, 30, 60):
            for threshold in (None, 0.0529112):
                for source in (4, 8, 12, 16, 20, 'hand'):
                    with self.subTest(fps=fps, threshold=threshold, source=source):
                        replay = self.replay(fps, threshold)
                        for sign in signs:
                            self.stroke(replay, sign, source)
                        replay.feed(1.2)
                        self.assertEqual(replay.types(), [
                            'SWIPE_UP' if sign < 0 else 'SWIPE_DOWN' for sign in signs])
                        self.assertTrue(all(b - a >= GESTURE_COOLDOWN - 1e-8
                                            for a, b in zip(replay.fire_times, replay.fire_times[1:])))

    def test_same_direction_short_pause_repeats(self):
        self.check_sequence((-1, -1, -1, -1))
        self.check_sequence((1, 1, 1, 1))

    def test_intentional_reverse_short_pause_repeats(self):
        self.check_sequence((-1, -1, 1, 1, -1))

    def test_new_stroke_is_tracked_before_cooldown_expires(self):
        for threshold in (None, 0.0529112):
            with self.subTest(threshold=threshold):
                replay = self.replay(60, threshold)
                self.stroke(replay, -1, 8, outward=0.12, back=0.08)
                self.assertEqual(replay.types(), ['SWIPE_UP'])
                self.assertEqual(replay.tracker.state, 'READY')
                self.assertLess(replay.now - replay.fire_times[0], GESTURE_COOLDOWN)
                self.assertTrue(all(sample.time > replay.fire_times[0]
                                    for sample in replay.tracker.history))
                replay.feed(0.10, dy=lambda t: -0.035 * t)
                self.assertEqual(replay.tracker.state, 'MOVING')
                self.assertEqual(replay.types(), ['SWIPE_UP'])
                replay.feed(0.08, dy=lambda t: -0.035 * (1 - t)).feed(1.2)
                self.assertEqual(replay.types(), ['SWIPE_UP', 'SWIPE_UP'])
                self.assertGreaterEqual(replay.fire_times[1] - replay.fire_times[0],
                                        GESTURE_COOLDOWN)

    def test_delayed_slow_return_stays_quiet(self):
        for fps in (15, 30, 60):
            for threshold in (None, 0.0529112):
                for source in (8, 'hand'):
                    with self.subTest(fps=fps, threshold=threshold, source=source):
                        replay = self.replay(fps, threshold)
                        axis = 'hand_y' if source == 'hand' else 'dy'
                        replay.feed(0.3, **{axis: lambda t: -0.035 * t})
                        replay.feed(0.75, **{axis: -0.035})
                        replay.feed(0.8, **{axis: lambda t: -0.035 * (1 - t)})
                        replay.feed(0.7)
                        self.assertEqual(replay.types(), ['SWIPE_UP'])
                        self.assertEqual(replay.tracker.state, 'READY')

    def test_rearmed_tremor_and_spikes_do_not_replay_consumed_stroke(self):
        for fps in (15, 30, 60):
            for threshold in (None, 0.0529112):
                with self.subTest(fps=fps, threshold=threshold):
                    replay = self.replay(fps, threshold)
                    self.stroke(replay, -1, 8)
                    replay.feed(2, dy=lambda t: 0.0015 * math.sin(t * 160))
                    for finger in (4, 8, 12, 16, 20):
                        spike = make_hand(fingers={finger: (0, -0.045)})
                        event = replay.tracker.update(spike, replay.now)
                        self.assertIsNone(event)
                        replay.now += 1 / fps
                        replay.feed(0.4)
                    self.assertEqual(replay.types(), ['SWIPE_UP'])
                    self.stroke(replay, 1, 8)
                    replay.feed(1.2)
                    self.assertEqual(replay.types(), ['SWIPE_UP', 'SWIPE_DOWN'])


if __name__ == '__main__':
    unittest.main()
