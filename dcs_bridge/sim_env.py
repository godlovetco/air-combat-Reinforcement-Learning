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

import math
import random
from typing import List, Optional, Tuple

import numpy as np

from . import geometry as geo

ARENA_XY = 200_000.0  # m
ARENA_Z = 11_000.0    # m

WIN_RANGE = 2_500.0   # m
WIN_ASPECT = 30.0     # deg

# Reactive-bandit limits (per decision step), matched to the agent's own
# 10 deg/step maneuver granularity so fights stay balanced and stable.
BANDIT_TURN = 10.0    # deg/step max heading change
BANDIT_GAMMA = 5.0    # deg/step max climb-angle change
OPPONENTS = ("straight", "pursuit", "evasive", "mixed")


class UCAVSimEnv:
    """Gym-style 1v1 environment (observation = legacy 72-dim input)."""

    def __init__(
        self,
        max_steps: int = 400,
        dt: float = 1.0,
        randomize: bool = True,
        shaping: float = 0.05,
        seed: Optional[int] = None,
        opponent: str = "straight",
    ):
        if opponent not in OPPONENTS:
            raise ValueError(f"unknown opponent {opponent!r}, expected {OPPONENTS}")
        self.max_steps = max_steps
        self.dt = dt
        self.randomize = randomize
        self.shaping = shaping
        self.opponent = opponent
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

        # Resolve "mixed" to a concrete behavior for this episode.
        self._episode_opponent = (
            self.rng.choice(("straight", "pursuit", "evasive"))
            if self.opponent == "mixed" else self.opponent
        )

        self.steps = 0
        self.done = False
        return self._obs()

    # ------------------------------------------------------------------ #
    def _update_bandit(self) -> None:
        """Reactive bandit: steer ``act_b`` toward its behavior's intent.

        ``straight`` leaves the bandit on its fixed profile (legacy default).
        ``pursuit`` turns to point at the agent; ``evasive`` breaks toward the
        beam when the agent is threatening from behind, else flies straight.
        """
        behavior = getattr(self, "_episode_opponent", "straight")
        if behavior == "straight":
            return

        dx = self.pos_r[0] - self.pos_b[0]
        dy = self.pos_r[1] - self.pos_b[1]
        dz = self.pos_r[2] - self.pos_b[2]
        horiz = math.hypot(dx, dy)
        d = math.hypot(horiz, dz)
        bearing = geo.wrap_heading(math.degrees(math.atan2(dx, dy)))
        v_b, gamma_b, psi_b = self.act_b

        if behavior == "pursuit":
            desired_psi = bearing
            desired_gamma = max(-20.0, min(20.0, math.degrees(math.atan2(dz, max(horiz, 1.0)))))
        else:  # evasive
            # Bandit aspect: angle between its velocity and the line of sight
            # back to the agent; large => the agent is in its rear hemisphere.
            feats = geo.situation(self.pos_b, self.act_b, self.pos_r, self.act_r)
            threatened = d < 8_000.0 and feats[0] > 90.0
            if threatened:
                # Break toward the beam (whichever 90 deg side is the nearer turn).
                left = geo.wrap_heading(bearing - 90.0)
                right = geo.wrap_heading(bearing + 90.0)
                desired_psi = (left if abs(geo.heading_error(left, psi_b))
                               <= abs(geo.heading_error(right, psi_b)) else right)
                desired_gamma = -5.0  # unload slightly to keep speed
            else:
                desired_psi, desired_gamma = psi_b, gamma_b

        turn = max(-BANDIT_TURN, min(BANDIT_TURN, geo.heading_error(desired_psi, psi_b)))
        psi_b = geo.wrap_heading(psi_b + turn)
        gamma_b = gamma_b + max(-BANDIT_GAMMA, min(BANDIT_GAMMA, desired_gamma - gamma_b))
        gamma_b = max(-geo.GAMMA_LIMIT_DEG, min(geo.GAMMA_LIMIT_DEG, gamma_b))
        self.act_b = [v_b, gamma_b, psi_b]

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
        self._update_bandit()  # reactive opponents adjust heading/climb here
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
