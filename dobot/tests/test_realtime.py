import threading
import time
import unittest

from dobot.sdk.realtime import run_control_loop


class RealtimeLoopTests(unittest.TestCase):
    def test_fixed_rate_without_catch_up_burst(self):
        stop = threading.Event()
        timestamps = []

        def step():
            timestamps.append(time.monotonic())

        timer = threading.Timer(0.22, stop.set)
        timer.start()
        stats = run_control_loop(step, stop, control_hz=25.0)
        timer.cancel()
        self.assertGreaterEqual(stats.ticks, 4)
        self.assertLessEqual(stats.deadlines_missed, 1)
        self.assertLess(max(
            b - a for a, b in zip(timestamps, timestamps[1:])), 0.06)

    def test_overrun_skips_missed_slots(self):
        stop = threading.Event()

        def step():
            time.sleep(0.045)
            calls.append(1)
            if len(calls) >= 2:
                stop.set()

        calls = []
        stats = run_control_loop(step, stop, control_hz=50.0)
        self.assertEqual(stats.ticks, 2)
        self.assertGreaterEqual(stats.deadlines_missed, 1)


if __name__ == "__main__":
    unittest.main()
