"""Point-mass 1v1 air-combat environment for training the UCAV policy.

A cleaned-up, importable rewrite of the legacy ``class_env.AirCombat``:

* red (the agent) picks one of 9 discrete maneuvers per second,
* blue (the bandit) flies its scripted profile (default: straight & level),
* the episode ends on a gun-envelope win/loss, boundary exit, or timeout.

Win condition (unchanged from the legacy project): inside 2500 m with our
aspect angle below 30 deg while the bandit's is above 30 deg and we hold an
altitude advantage -- i.e. converted to the bandit's six o'clock.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from . import geometry as geo

ARENA_XY = 200_000.0  # m
ARENA_Z = 11_000.0    # m

WIN_RANGE = 2_500.0   # m
WIN_ASPECT = 30.0     # deg


class UCAVSimEnv:
    """Gym-style 1v1 environment (observation = legacy 72-dim input)."""

    def __init__(
        self,
        max_steps: int = 400,
        dt: float = 1.0,
        randomize: bool = True,
        shaping: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.max_steps = max_steps
        self.dt = dt
        self.randomize = randomize
        self.shaping = shaping
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        # Legacy head-on setup: 10 km apart, co-altitude, closing.
        self.pos_r = [130_000.0, 100_000.0, 3_000.0]
        self.pos_b = [130_000.0, 110_000.0, 3_000.0]
        self.act_r = [250.0, 0.0, 0.0]     # v, gamma, psi (north)
        self.act_b = [250.0, 0.0, 180.0]   # v, gamma, psi (south, toward us)

        if self.randomize:
            r = self.rng
            self.pos_b[0] += r.uniform(-4_000.0, 4_000.0)
            self.pos_b[1] += r.uniform(-2_000.0, 6_000.0)
            self.pos_b[2] += r.uniform(-800.0, 800.0)
            self.act_b[2] = geo.wrap_heading(180.0 + r.uniform(-30.0, 30.0))
            self.act_r[2] = geo.wrap_heading(r.uniform(-20.0, 20.0))

        self.steps = 0
        self.done = False
        return self._obs()

    def _obs(self) -> np.ndarray:
        return geo.build_network_input(
            self.pos_r, self.act_r, self.pos_b, self.act_b, self.dt
        )

    # ------------------------------------------------------------------ #
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        cands = geo.candidate_actions(*self.act_r)
        self.act_r = cands[action_idx]
        self.pos_r = geo.step_point_mass(self.pos_r, self.act_r, self.dt)
        self.pos_b = geo.step_point_mass(self.pos_b, self.act_b, self.dt)
        self.steps += 1

        feats = geo.situation(self.pos_r, self.act_r, self.pos_b, self.act_b)
        q_r, q_b, d, _, delta_h = feats[0], feats[1], feats[2], feats[3], feats[4]

        reward = 0.0
        info = {"outcome": None, "q_r": q_r, "q_b": q_b, "d": d}

        if self.shaping:
            # Dense shaping in the spirit of Yoo/Kim/Shim (ICCAS 2021),
            # Table 2: an aspect-angle term plus a closing-distance term.
            reward += self.shaping * (q_b - q_r) / 180.0
            if d < 10_000.0:
                reward += 0.5 * self.shaping * (1.0 - d / 10_000.0)

        if d < WIN_RANGE and q_r < WIN_ASPECT and q_b > WIN_ASPECT and delta_h > 100.0:
            reward += 10.0
            self.done = True
            info["outcome"] = "win"
        elif d < WIN_RANGE and q_r > WIN_ASPECT and q_b < WIN_ASPECT and delta_h < 100.0:
            reward -= 10.0
            self.done = True
            info["outcome"] = "loss"
        elif self._out_of_bounds():
            reward -= 5.0
            self.done = True
            info["outcome"] = "out_of_bounds"
        elif self.steps >= self.max_steps:
            reward -= 5.0
            self.done = True
            info["outcome"] = "timeout"

        return self._obs(), reward, self.done, info

    def _out_of_bounds(self) -> bool:
        for pos in (self.pos_r, self.pos_b):
            if not (0.0 < pos[0] < ARENA_XY and 0.0 < pos[1] < ARENA_XY):
                return True
            if not (100.0 < pos[2] < ARENA_Z):
                return True
        return False

    # ------------------------------------------------------------------ #
    def trajectory_header(self) -> List[str]:
        return ["x_r", "y_r", "z_r", "x_b", "y_b", "z_b"]

    def trajectory_row(self) -> List[float]:
        return [*self.pos_r, *self.pos_b]
