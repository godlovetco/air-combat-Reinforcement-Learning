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

# --------------------------------------------------------------------- #
# Energy action set (opt-in)
# --------------------------------------------------------------------- #
# The legacy action set holds speed fixed, so there is no energy game: every
# turn is free and a fight between equals decays into a pure angles problem.
# The energy set adds a throttle axis (burner / hold / idle) to each of the 9
# maneuvers, and couples speed to the flight path the way a real jet is
# coupled -- gravity along the climb angle, induced drag in the turn.
THROTTLE_DELTAS: Tuple[float, ...] = (+1.0, 0.0, -1.0)  # burner, hold, idle
ACTION_DELTAS_ENERGY: Tuple[Tuple[float, float, float], ...] = tuple(
    (d_gamma, d_psi, d_thr)
    for d_gamma, d_psi in ACTION_DELTAS
    for d_thr in THROTTLE_DELTAS
)

ACTION_SETS = {"legacy": ACTION_DELTAS, "energy": ACTION_DELTAS_ENERGY}
DEFAULT_ACTION_SET = "legacy"

V_MIN = 120.0           # m/s, below this the jet is out of usable energy
V_MAX = 400.0           # m/s
THROTTLE_ACCEL = 4.0    # m/s^2 commanded by full burner / idle
G_ACCEL = 9.81          # m/s^2
TURN_BLEED = 3.0        # m/s^2 of induced drag at the full 10 deg/step turn
V_CORNER = 200.0        # m/s, above which the full climb angle is available


def max_climb_angle(v: float) -> float:
    """Climb angle the jet can hold at speed ``v`` (energy action set only).

    Without this a point mass with a hard speed floor can zoom-climb forever:
    it trades speed for altitude, bottoms out at ``V_MIN``, and keeps going up
    at ``V_MIN`` indefinitely.  Measured consequence in a self-play fight: every
    single unresolved engagement ended at the 11 km ceiling, the bandit five
    times more often than the agent.  Real jets run out of energy and the nose
    falls, so climb authority scales from the full limit at corner speed down to
    level flight at ``V_MIN``.  Descending is never limited -- unloading is how
    you get the energy back.
    """
    frac = (v - V_MIN) / (V_CORNER - V_MIN)
    return GAMMA_LIMIT_DEG * max(0.0, min(1.0, frac))


def action_set_dims(action_set: str = DEFAULT_ACTION_SET) -> Tuple[int, int]:
    """``(num_actions, input_dim)`` for a named action set."""
    if action_set not in ACTION_SETS:
        raise ValueError(
            f"unknown action set {action_set!r}, expected {tuple(ACTION_SETS)}"
        )
    n = len(ACTION_SETS[action_set])
    return n, n * STATE_DIM


def action_set_for(num_actions: int) -> str:
    """Name the action set with this many actions (used to read a checkpoint)."""
    for name, deltas in ACTION_SETS.items():
        if len(deltas) == num_actions:
            return name
    raise ValueError(f"no action set has {num_actions} actions")

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


def energy_step(
    v: float, new_gamma_deg: float, d_psi: float, throttle: float, dt: float = 1.0
) -> float:
    """Speed after one step of the energy model, clamped to the usable band.

    ``dv/dt = throttle*THROTTLE_ACCEL - g*sin(gamma) - induced drag``: burner
    buys speed, climbing spends it, and a hard turn bleeds it.  This is the
    coupling that makes an energy fight an energy fight -- pull hard and you
    slow down, unload and you get it back.
    """
    accel = (
        throttle * THROTTLE_ACCEL
        - G_ACCEL * math.sin(new_gamma_deg * DEG)
        - TURN_BLEED * abs(d_psi) / 10.0
    )
    return max(V_MIN, min(V_MAX, v + accel * dt))


def candidate_actions(
    v: float,
    gamma_deg: float,
    psi_deg: float,
    action_set: str = DEFAULT_ACTION_SET,
    dt: float = 1.0,
) -> List[List[float]]:
    """The candidate kinematic commands from the current command state.

    Climb angle is clamped to +/-GAMMA_LIMIT_DEG and heading wrapped, so the
    command state cannot run away after many decisions.  ``legacy`` (the
    default) returns the 9 constant-speed maneuvers of the original project;
    ``energy`` returns 27 -- the same 9 crossed with burner/hold/idle -- and
    propagates speed through ``energy_step``.
    """
    if action_set not in ACTION_SETS:
        raise ValueError(
            f"unknown action set {action_set!r}, expected {tuple(ACTION_SETS)}"
        )
    out = []
    # Climb authority is bounded by the energy on hand; diving never is.
    up_limit = GAMMA_LIMIT_DEG if action_set == "legacy" else max_climb_angle(v)
    for delta in ACTION_SETS[action_set]:
        d_gamma, d_psi = delta[0], delta[1]
        new_gamma = max(-GAMMA_LIMIT_DEG, min(up_limit, gamma_deg + d_gamma))
        new_v = v if len(delta) == 2 else energy_step(v, new_gamma, d_psi, delta[2], dt)
        out.append([new_v, new_gamma, wrap_heading(psi_deg + d_psi)])
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

# The legacy scale of 1.0 on ``delta_v2`` was harmless only because the legacy
# action set pins both aircraft at their starting speed, making that feature
# identically zero in every situation it ever saw.  With a throttle axis it
# ranges over +/-145,000 while every other normalized feature is order 1, which
# would swamp the first layer.  The energy set therefore scales it like ``v2``.
_NORM_ENERGY = np.array(
    [200.0, 200.0, 20000.0, 200.0, 10000.0, 40000.0, 40000.0, 10000.0]
)
_NORMS = {"legacy": _NORM, "energy": _NORM_ENERGY}


def normalize(
    features: Sequence[float], action_set: str = DEFAULT_ACTION_SET
) -> np.ndarray:
    """Scale an 8-feature situation vector for the given action set."""
    if action_set not in _NORMS:
        raise ValueError(
            f"unknown action set {action_set!r}, expected {tuple(_NORMS)}"
        )
    return np.asarray(features, dtype=np.float64) / _NORMS[action_set]


def build_network_input(
    pos_r: Sequence[float],
    act_r: Sequence[float],
    pos_b: Sequence[float],
    act_b: Sequence[float],
    dt: float = 1.0,
    next_pos_b: Optional[Sequence[float]] = None,
    action_set: str = DEFAULT_ACTION_SET,
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
    cands = candidate_actions(act_r[0], act_r[1], act_r[2], action_set, dt)
    next_b = list(next_pos_b) if next_pos_b is not None else step_point_mass(pos_b, act_b, dt)
    rows = []
    for cand in cands:
        next_r = step_point_mass(pos_r, cand, dt)
        rows.append(normalize(situation(next_r, cand, next_b, act_b), action_set))
    return np.concatenate(rows)
