"""Bank-to-turn inner-loop controller.

The RL policy decides *what* to do every half second or so: a target climb
angle ``gamma_cmd``, target heading ``psi_cmd`` and target speed ``v_cmd``.
This module decides *how* to do it at telemetry rate (~20 Hz), converting
those targets into DCS joystick axis values:

    pitch, roll, rudder in [-1, 1]   and   thrust in [0, 1]

The sign conventions of the DCS axis commands differ per airframe/setup, so
they are configurable in :class:`AutopilotConfig` (see also the constants at
the top of the Lua export script).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import heading_error


@dataclass
class AutopilotConfig:
    # Outer loop: heading error -> bank target
    bank_per_heading_deg: float = 2.5     # deg of bank per deg of heading error
    max_bank_deg: float = 70.0

    # Roll loop: bank error -> aileron
    roll_p: float = 0.03                  # stick per deg of bank error
    roll_d: float = 0.012                 # damping on bank rate (per deg/s)

    # Pitch loop: pitch-attitude error -> elevator
    pitch_p: float = 0.05
    pitch_d: float = 0.02                 # damping on pitch rate (per deg/s)
    pitch_trim: float = 0.03              # steady-state pull for level flight
    bank_compensation: float = 0.35       # extra pull with bank (turn compensation)
    max_pitch_cmd_deg: float = 45.0

    # Yaw: small coordination rudder from heading error
    rudder_p: float = 0.004
    max_rudder: float = 0.3

    # Speed loop: TAS error -> throttle
    throttle_p: float = 0.02              # throttle per m/s of speed error
    throttle_base: float = 0.75

    # Axis direction fixes. DCS: positive pitch axis = stick forward (nose
    # down) on most modules, hence the default inversion.
    invert_pitch: bool = True
    invert_roll: bool = False
    invert_rudder: bool = False


@dataclass
class Controls:
    pitch: float = 0.0
    roll: float = 0.0
    rudder: float = 0.0
    thrust: float = 0.5   # 0..1
    trigger: int = 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Autopilot:
    cfg: AutopilotConfig = field(default_factory=AutopilotConfig)
    _last_bank: float = 0.0
    _last_pitch: float = 0.0
    _last_t: float = 0.0

    def command(
        self,
        t: float,
        pitch_deg: float,
        bank_deg: float,
        heading_deg: float,
        tas: float,
        gamma_cmd_deg: float,
        psi_cmd_deg: float,
        v_cmd: float,
    ) -> Controls:
        """One control step from current attitude toward the maneuver targets."""
        cfg = self.cfg
        dt = t - self._last_t if self._last_t and t > self._last_t else 0.05
        bank_rate = (bank_deg - self._last_bank) / dt
        pitch_rate = (pitch_deg - self._last_pitch) / dt
        self._last_bank, self._last_pitch, self._last_t = bank_deg, pitch_deg, t

        # --- lateral: heading error -> bank target -> aileron -------------
        e_psi = heading_error(psi_cmd_deg, heading_deg)
        bank_cmd = _clamp(cfg.bank_per_heading_deg * e_psi, -cfg.max_bank_deg, cfg.max_bank_deg)
        roll = cfg.roll_p * (bank_cmd - bank_deg) - cfg.roll_d * bank_rate

        # --- longitudinal: climb-angle target -> pitch target -> elevator -
        # Pitch attitude ~ flight-path angle + a small AoA margin; add pull
        # proportional to (1 - cos(bank)) so turns do not dump the nose.
        pitch_cmd = _clamp(gamma_cmd_deg + 2.0, -cfg.max_pitch_cmd_deg, cfg.max_pitch_cmd_deg)
        turn_pull = cfg.bank_compensation * (1.0 - math.cos(math.radians(bank_deg)))
        elevator = (
            cfg.pitch_p * (pitch_cmd - pitch_deg)
            - cfg.pitch_d * pitch_rate
            + cfg.pitch_trim
            + turn_pull
        )

        rudder = _clamp(cfg.rudder_p * e_psi, -cfg.max_rudder, cfg.max_rudder)
        throttle = _clamp(cfg.throttle_base + cfg.throttle_p * (v_cmd - tas), 0.0, 1.0)

        pitch_axis = _clamp(elevator, -1.0, 1.0)
        roll_axis = _clamp(roll, -1.0, 1.0)
        if cfg.invert_pitch:
            pitch_axis = -pitch_axis
        if cfg.invert_roll:
            roll_axis = -roll_axis
        if cfg.invert_rudder:
            rudder = -rudder

        return Controls(pitch=pitch_axis, roll=roll_axis, rudder=rudder, thrust=throttle)
