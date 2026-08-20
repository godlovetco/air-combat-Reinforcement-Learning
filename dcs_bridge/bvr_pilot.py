"""Scripted BVR pilot: the timeline, flown.

This exists because it wins.  Against every reactive bandit in
:mod:`dcs_bridge.bvr_env`, over 400 engagements on each of two held-out seeds,
the twenty lines below beat every reinforcement-learning policy trained so far
on every axis -- 0.94 against 0.61 on a straight bandit, 0.99 against 0.80 on a
pursuing one.  Shipping the learned policy as the BVR pilot would mean shipping
the worse of the two because it is the more interesting one.

The timeline has three states and they are strictly ordered, which is the whole
trick:

1. **Defend.**  A missile guiding on you outranks everything.  Turn to put it
   on the 3/9 line and hold it there -- and beam the *missile*, not the
   aircraft that fired it, because those bearings diverge as the round closes
   and beaming the shooter walks you out of the seeker's Doppler gate while it
   feels like you are notching.
2. **Support.**  Your own shot is inertial until its seeker goes active, so
   until then the radar has to stay on the target.  Crank -- turn as far off
   the target as the antenna will tolerate -- to open the range without
   throwing the shot away.
3. **Commit.**  Nothing in the air: point at the target and close.

:class:`TimelinePilot` consumes the same situation block the WSO advisor reads
(:func:`dcs_bridge.bvr_env.wso_situation`), so it works from telemetry rather
than from the environment's internals and can be driven by a live DCS feed.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from . import geometry as geo

CRANK_DEG = 50.0        # off-boresight a supporting shooter can hold
NOTCH_DEG = 90.0        # put the threat on the 3/9 line
LEVEL_BIAS = True       # break heading ties toward the flattest climb angle


def _bearing(from_pos: Sequence[float], to_pos: Sequence[float]) -> float:
    return geo.wrap_heading(math.degrees(math.atan2(
        to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])))


def _nearer_beam(bearing_deg: float, psi_deg: float) -> float:
    """Whichever 90-degree offset from a bearing is the shorter turn."""
    left = geo.wrap_heading(bearing_deg - NOTCH_DEG)
    right = geo.wrap_heading(bearing_deg + NOTCH_DEG)
    return (left if abs(geo.heading_error(left, psi_deg))
            <= abs(geo.heading_error(right, psi_deg)) else right)


class TimelinePilot:
    """Scripted BVR maneuver policy.  Same ``act`` surface as ``QNetwork``."""

    def __init__(self, crank_deg: float = CRANK_DEG):
        self.crank_deg = crank_deg

    # ------------------------------------------------------------------ #
    def desired_heading(self, sit: dict) -> float:
        """The heading the timeline wants, given a WSO situation block.

        ``sit`` needs ``own`` (heading_deg), ``target_bearing_deg`` and the
        ``bvr`` block.
        """
        psi = float(sit["own"]["heading_deg"])
        hot = float(sit["target_bearing_deg"])
        bvr = sit.get("bvr") or {}

        threat = bvr.get("threat")
        if threat is not None and threat.get("active"):
            # 1 - defend: beam the missile, tracked every step.
            return _nearer_beam(
                geo.wrap_heading(psi + float(threat["off_nose_deg"])), psi)

        if bvr.get("shot_in_flight") and float(bvr.get("support_owed_s", 0.0)) > 0.0:
            # 2 - support: crank the shortest way that keeps the target inside
            # the antenna, rather than always cranking the same direction.
            left = geo.wrap_heading(hot - self.crank_deg)
            right = geo.wrap_heading(hot + self.crank_deg)
            return (left if abs(geo.heading_error(left, psi))
                    <= abs(geo.heading_error(right, psi)) else right)

        return hot  # 3 - commit

    # ------------------------------------------------------------------ #
    def act(self, sit: dict, action_set: str = geo.DEFAULT_ACTION_SET,
            dt: float = 1.0) -> int:
        """Index of the candidate maneuver closest to the desired heading."""
        want = self.desired_heading(sit)
        own = sit["own"]
        cands = geo.candidate_actions(
            float(own.get("tas", 250.0)), float(own.get("gamma_deg", 0.0)),
            float(own["heading_deg"]), action_set=action_set, dt=dt)
        return min(
            range(len(cands)),
            key=lambda i: (abs(geo.heading_error(want, cands[i][2])),
                           abs(cands[i][1]) if LEVEL_BIAS else 0.0),
        )


def situation_from_env(env, who: str = "agent") -> dict:
    """Build a ``TimelinePilot`` situation from a :class:`BVRSimEnv`.

    Only for evaluation and tests -- in DCS the same block comes from telemetry.
    """
    from .bvr_env import wso_situation

    own_pos = env.pos_r if who == "agent" else env.pos_b
    own_act = env.act_r if who == "agent" else env.act_b
    tgt_pos = env.pos_b if who == "agent" else env.pos_r
    return {
        "own": {"heading_deg": own_act[2], "gamma_deg": own_act[1],
                "tas": own_act[0], "altitude_m": own_pos[2]},
        "target_bearing_deg": _bearing(own_pos, tgt_pos),
        "bvr": wso_situation(env, who),
    }
