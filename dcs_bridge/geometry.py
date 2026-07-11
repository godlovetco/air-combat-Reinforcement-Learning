"""Combat geometry shared by the training simulator and the DCS bridge.

Conventions (same as the legacy ``class_env.py``):

* East-North-Up (ENU) coordinates: ``x`` east, ``y`` north, ``z`` up (meters).
* ``psi``   -- compass heading in degrees (0 = north, 90 = east).
* ``gamma`` -- flight-path (climb) angle in degrees (positive = climbing).
* An "action" is the kinematic command triple ``(v, gamma, psi)``.

The situation ("taishi") feature vector has 8 entries, identical to the
legacy environment::

    [q_r, q_b, d, beta, delta_h, delta_v2, v2, h]

    q_r      deg   own aspect angle: angle between our velocity vector and
                   the line of sight to the bandit (0 = nose-on the bandit)
    q_b      deg   bandit aspect angle: angle between the bandit's velocity
                   and the line of sight back to us (0 = bandit nose-on us)
    d        m     slant range
    beta     deg   angle between the two velocity vectors
    delta_h  m     altitude advantage (own - bandit)
    delta_v2 m2/s2 v_r^2 - v_b^2
    v2       m2/s2 v_r^2
    h        m     own altitude

Note: the legacy ``generate_state`` forgot the degree->radian conversion
when computing ``q_b``.  This module computes ``q_b`` correctly; networks
trained with this package are consistent with the fixed formula.

The network input is the legacy 72-vector: for each of the 9 candidate
maneuvers, propagate both aircraft one decision step and concatenate the 9
normalized situation vectors.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEG = math.pi / 180.0

# Candidate maneuvers, legacy ordering: (d_gamma, d_psi) in degrees.
ACTION_DELTAS: Tuple[Tuple[float, float], ...] = (
    (+10.0, +10.0),  # 0: climb harder, turn right
    (+10.0, 0.0),    # 1: climb harder, hold heading
    (+10.0, -10.0),  # 2: climb harder, turn left
    (0.0, +10.0),    # 3: hold climb, turn right
    (0.0, 0.0),      # 4: hold everything
    (0.0, -10.0),    # 5: hold climb, turn left
    (-10.0, +10.0),  # 6: descend, turn right
    (-10.0, 0.0),    # 7: descend, hold heading
    (-10.0, -10.0),  # 8: descend, turn left
)

NUM_ACTIONS = len(ACTION_DELTAS)
STATE_DIM = 8
INPUT_DIM = NUM_ACTIONS * STATE_DIM  # 72, as in the legacy network

GAMMA_LIMIT_DEG = 70.0  # keep away from the vertical singularity


def wrap_heading(psi_deg: float) -> float:
    """Wrap a heading to [0, 360)."""
    return psi_deg % 360.0


def heading_error(target_deg: float, current_deg: float) -> float:
    """Shortest signed heading difference in degrees, in [-180, 180)."""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def velocity_enu(v: float, gamma_deg: float, psi_deg: float) -> Tuple[float, float, float]:
    """ENU velocity components for speed/climb-angle/heading."""
    g = gamma_deg * DEG
    p = psi_deg * DEG
    return (
        v * math.cos(g) * math.sin(p),  # east
        v * math.cos(g) * math.cos(p),  # north
        v * math.sin(g),                # up
    )


def step_point_mass(
    pos: Sequence[float], action: Sequence[float], dt: float = 1.0
) -> List[float]:
    """Propagate a point-mass aircraft one step of ``dt`` seconds."""
    vx, vy, vz = velocity_enu(action[0], action[1], action[2])
    return [pos[0] + vx * dt, pos[1] + vy * dt, pos[2] + vz * dt]


def candidate_actions(
    v: float, gamma_deg: float, psi_deg: float
) -> List[List[float]]:
    """The 9 candidate kinematic commands from the current command state.

    Climb angle is clamped to +/-GAMMA_LIMIT_DEG and heading wrapped, so the
    command state cannot run away after many decisions.
    """
    out = []
    for d_gamma, d_psi in ACTION_DELTAS:
        new_gamma = max(-GAMMA_LIMIT_DEG, min(GAMMA_LIMIT_DEG, gamma_deg + d_gamma))
        out.append([v, new_gamma, wrap_heading(psi_deg + d_psi)])
    return out


def _clamped_acos_deg(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x))) / DEG


def situation(
    pos_r: Sequence[float],
    act_r: Sequence[float],
    pos_b: Sequence[float],
    act_b: Sequence[float],
) -> List[float]:
    """8-feature situation vector [q_r, q_b, d, beta, delta_h, delta_v2, v2, h]."""
    dx = pos_b[0] - pos_r[0]
    dy = pos_b[1] - pos_r[1]
    dz = pos_b[2] - pos_r[2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    d = max(d, 1e-6)

    v_r, gamma_r, psi_r = act_r
    v_b, gamma_b, psi_b = act_b
    ur = velocity_enu(1.0, gamma_r, psi_r)
    ub = velocity_enu(1.0, gamma_b, psi_b)

    q_r = _clamped_acos_deg((dx * ur[0] + dy * ur[1] + dz * ur[2]) / d)
    q_b = _clamped_acos_deg((-dx * ub[0] - dy * ub[1] - dz * ub[2]) / d)
    beta = _clamped_acos_deg(ur[0] * ub[0] + ur[1] * ub[1] + ur[2] * ub[2])

    delta_h = pos_r[2] - pos_b[2]
    delta_v2 = v_r ** 2 - v_b ** 2
    return [q_r, q_b, d, beta, delta_h, delta_v2, v_r ** 2, pos_r[2]]


# Legacy normalization constants (kept identical so behavior is comparable).
_NORM = np.array([200.0, 200.0, 20000.0, 200.0, 10000.0, 1.0, 40000.0, 10000.0])


def normalize(features: Sequence[float]) -> np.ndarray:
    """Scale an 8-feature situation vector, same constants as class_env."""
    return np.asarray(features, dtype=np.float64) / _NORM


def build_network_input(
    pos_r: Sequence[float],
    act_r: Sequence[float],
    pos_b: Sequence[float],
    act_b: Sequence[float],
    dt: float = 1.0,
    next_pos_b: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """72-dim network input: normalized situations after each candidate maneuver.

    For every one of the 9 candidate commands, both aircraft are propagated
    ``dt`` seconds (the bandit holds its current command) and the resulting
    situation is normalized and concatenated -- exactly the legacy scheme of
    feeding "what each action would lead to" into the value network.

    ``next_pos_b`` overrides the straight-line bandit propagation with an
    externally predicted position (see ``predictor.TurnRatePredictor``), the
    trajectory-prediction idea of Yoo/Kim/Shim (ICCAS 2021).
    """
    cands = candidate_actions(act_r[0], act_r[1], act_r[2])
    next_b = list(next_pos_b) if next_pos_b is not None else step_point_mass(pos_b, act_b, dt)
    rows = []
    for cand in cands:
        next_r = step_point_mass(pos_r, cand, dt)
        rows.append(normalize(situation(next_r, cand, next_b, act_b)))
    return np.concatenate(rows)
