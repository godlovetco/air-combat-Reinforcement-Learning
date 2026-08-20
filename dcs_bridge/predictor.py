"""Target trajectory prediction for lead pursuit.

Yoo, Kim & Shim, "Deep Reinforcement Learning based Autonomous Air-to-Air
Combat using Target Trajectory Prediction" (ICCAS 2021) validated an AI
pilot in DCS World and showed that predicting the bandit's future positions
(they sampled 10 positions over the next 4 seconds with a Seq2Seq-LSTM) is
what turns tail-chasing into interception.

This module keeps that idea but stays dependency-free: the bandit's recent
positions are fit to a constant-speed, constant-turn-rate, constant-climb
model, which is the standard motion model for maneuvering aircraft over
1-4 second horizons and needs no trained weights.
"""

from __future__ import annotations

import collections
import math
from typing import Deque, List, Optional, Sequence, Tuple

from .geometry import wrap_heading

Vec3 = Tuple[float, float, float]

PREDICT_HORIZON = 4.0   # s, matches the paper's 80-step / 4 s window
PREDICT_SAMPLES = 10    # positions sampled over the horizon, as in the paper


class TurnRatePredictor:
    """Constant-turn-rate bandit motion model built from position history."""

    def __init__(self, history_seconds: float = 2.5):
        self.history_seconds = history_seconds
        self.samples: Deque[Tuple[float, Vec3]] = collections.deque()

    def update(self, t: float, pos: Sequence[float]) -> None:
        p = (float(pos[0]), float(pos[1]), float(pos[2]))
        if self.samples and t <= self.samples[-1][0]:
            return
        self.samples.append((t, p))
        while self.samples and t - self.samples[0][0] > self.history_seconds:
            self.samples.popleft()

    # ------------------------------------------------------------------ #
    def motion_state(self) -> Optional[Tuple[Vec3, float, float, float, float]]:
        """(position, ground speed, heading deg, turn rate deg/s, climb rate)."""
        if len(self.samples) < 3:
            return None
        t2, p2 = self.samples[-1]
        mid = len(self.samples) // 2
        t1, p1 = self.samples[mid]
        t0, p0 = self.samples[0]
        if t2 - t1 < 1e-3 or t1 - t0 < 1e-3:
            return None

        def horiz_velocity(pa: Vec3, pb: Vec3, dt: float) -> Tuple[float, float, float]:
            return ((pb[0] - pa[0]) / dt, (pb[1] - pa[1]) / dt, (pb[2] - pa[2]) / dt)

        vx1, vy1, vz1 = horiz_velocity(p0, p1, t1 - t0)
        vx2, vy2, vz2 = horiz_velocity(p1, p2, t2 - t1)
        speed = math.hypot(vx2, vy2)
        if speed < 10.0:
            return None
        hdg1 = math.degrees(math.atan2(vx1, vy1))
        hdg2 = math.degrees(math.atan2(vx2, vy2))
        d_hdg = (hdg2 - hdg1 + 180.0) % 360.0 - 180.0
        dt_mid = (t2 + t1) / 2.0 - (t1 + t0) / 2.0
        turn_rate = d_hdg / dt_mid if dt_mid > 1e-3 else 0.0
        turn_rate = max(-25.0, min(25.0, turn_rate))  # physically plausible
        return (p2, speed, wrap_heading(hdg2), turn_rate, vz2)

    def predict(self, horizon: float) -> Optional[Vec3]:
        """Predicted position ``horizon`` seconds after the last update."""
        state = self.motion_state()
        if state is None:
            return None
        (px, py, pz), speed, hdg, rate, climb = state
        h0 = math.radians(hdg)
        if abs(rate) < 0.1:
            dx = speed * math.sin(h0) * horizon
            dy = speed * math.cos(h0) * horizon
        else:
            w = math.radians(rate)
            h1 = h0 + w * horizon
            # Integral of speed*sin/cos(h0 + w*t) over the horizon (arc).
            dx = speed / w * (math.cos(h0) - math.cos(h1))
            dy = speed / w * (math.sin(h1) - math.sin(h0))
        return (px + dx, py + dy, pz + climb * horizon)

    def sample_track(
        self,
        horizon: float = PREDICT_HORIZON,
        n: int = PREDICT_SAMPLES,
    ) -> List[Vec3]:
        """The paper-style list of n predicted positions over the horizon."""
        out = []
        for i in range(1, n + 1):
            p = self.predict(horizon * i / n)
            if p is None:
                return []
            out.append(p)
        return out

    def intercept_point(
        self,
        own_pos: Sequence[float],
        own_speed: float,
        max_horizon: float = PREDICT_HORIZON,
    ) -> Optional[Vec3]:
        """Lead point: predicted bandit position at our estimated flight time.

        Picks, among the sampled future positions, the one whose flight time
        from us (at ``own_speed``) best matches its prediction horizon --
        the same selection rule the paper applies to its 10 predicted states.
        """
        track = self.sample_track(max_horizon)
        if not track:
            return None
        own_speed = max(own_speed, 50.0)
        best, best_err = None, float("inf")
        for i, p in enumerate(track, start=1):
            horizon = max_horizon * i / len(track)
            flight_time = math.dist(own_pos, p) / own_speed
            err = abs(flight_time - horizon)
            if err < best_err:
                best, best_err = p, err
        return best
