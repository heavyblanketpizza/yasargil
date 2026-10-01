"""Replay clocks. ``SimulatedClock`` jumps to each frame (as fast as possible and
deterministic); ``RealClock`` follows wall time scaled by ``speed``."""
from __future__ import annotations

import time


class SimulatedClock:
    def __init__(self, start_ms=0):
        self._now = float(start_ms)

    def now_ms(self):
        return self._now

    def sleep_until(self, t_ms):
        self._now = max(self._now, float(t_ms))

    def advance(self, ms):
        self._now += ms


class RealClock:
    def __init__(self, speed=1.0):
        if not speed > 0:
            raise ValueError("speed must be positive")
        self.speed = speed
        self._start = time.monotonic()

    def now_ms(self):
        return (time.monotonic() - self._start) * 1000 * self.speed

    def sleep_until(self, t_ms):
        delay = (t_ms - self.now_ms()) / 1000 / self.speed
        if delay > 0:
            time.sleep(delay)


def make_clock(speed):
    """``speed`` 0 replays as fast as possible; otherwise in scaled real time."""
    if speed < 0:
        raise ValueError("speed must be zero (as fast as possible) or positive")
    return SimulatedClock() if speed == 0 else RealClock(speed)


def wall_ms():
    return time.perf_counter() * 1000
