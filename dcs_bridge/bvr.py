"""Beyond-visual-range engagement model: radar, weapon envelope, missiles.

The rest of this package models a gun fight -- close the range, get inside 30
degrees of aspect, take the shot.  A BVR fight is a different problem and is
decided long before that: whether your radar holds the lock, whether the shot
is inside the target's escape window, and whether you can keep supporting your
own missile while defeating theirs.

This module is the physics of that fight, deliberately kept separate from the
learning code so it can be read, tested and corrected on its own:

* :class:`Radar` -- gimbal limit, range, and the Doppler notch that a target
  beaming the antenna disappears into.
* :func:`kinematic_range` / :func:`launch_authority` -- how far a missile
  actually reaches given altitude and whether the target is closing or running,
  and the Rmax / Rne / Rmin bands that decide when a shot is worth taking.
* :class:`Missile` -- a guided round in flight: inertial on datalink until its
  own seeker goes active ("pitbull"), defeatable by breaking the supporting
  lock before then, by notching after, or simply by outrunning its energy.

Ranges and timings are representative of a modern active-radar missile
(AIM-120C class) rather than exact: DCS itself does not publish these numbers,
and the goal is a fight with the right *shape* -- crank, notch, drag, F-pole --
not a ballistics table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import geometry as geo

# --------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------- #
RADAR_GIMBAL_DEG = 60.0      # antenna scan limit either side of the nose
RADAR_RANGE = 90_000.0       # m, detection range against a fighter-size target
NOTCH_CLOSURE = 50.0         # m/s of *target* radial velocity below which the
                             # Doppler filter rejects the return as ground
                             # clutter -- this is what "notching" exploits


def line_of_sight(pos_from: Sequence[float], pos_to: Sequence[float]):
    """``(unit vector, range)`` from one point to another."""
    dx = pos_to[0] - pos_from[0]
    dy = pos_to[1] - pos_from[1]
    dz = pos_to[2] - pos_from[2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d < 1e-6:
        return (0.0, 0.0, 0.0), 0.0
    return (dx / d, dy / d, dz / d), d


def closure_rate(
    pos_a: Sequence[float], act_a: Sequence[float],
    pos_b: Sequence[float], act_b: Sequence[float],
) -> float:
    """Range rate along the line of sight, positive when closing (m/s)."""
    los, d = line_of_sight(pos_a, pos_b)
    if d == 0.0:
        return 0.0
    va = geo.velocity_enu(*act_a)
    vb = geo.velocity_enu(*act_b)
    rel = (va[0] - vb[0], va[1] - vb[1], va[2] - vb[2])
    return rel[0] * los[0] + rel[1] * los[1] + rel[2] * los[2]


def radial_speed(
    pos_from: Sequence[float], pos_to: Sequence[float], act_to: Sequence[float],
) -> float:
    """The target's own velocity component along the line of sight (m/s).

    This, not the total closure, is what a pulse-Doppler radar filters on.  A
    target flying perpendicular to the line of sight -- "beaming", or in the
    notch -- has a radial velocity near zero and is thrown away with the ground
    clutter no matter how fast the shooter is closing.
    """
    los, d = line_of_sight(pos_from, pos_to)
    if d == 0.0:
        return 0.0
    v = geo.velocity_enu(*act_to)
    return v[0] * los[0] + v[1] * los[1] + v[2] * los[2]


def off_boresight(
    pos_from: Sequence[float], act_from: Sequence[float],
    pos_to: Sequence[float],
) -> float:
    """Angle between the nose and the target, in degrees."""
    los, d = line_of_sight(pos_from, pos_to)
    if d == 0.0:
        return 0.0
    nose = geo.velocity_enu(1.0, act_from[1], act_from[2])
    dot = nose[0] * los[0] + nose[1] * los[1] + nose[2] * los[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


@dataclass(frozen=True)
class Radar:
    gimbal_deg: float = RADAR_GIMBAL_DEG
    max_range: float = RADAR_RANGE
    notch_closure: float = NOTCH_CLOSURE

    def can_see(
        self,
        pos_own: Sequence[float], act_own: Sequence[float],
        pos_tgt: Sequence[float], act_tgt: Sequence[float],
    ) -> bool:
        """Whether the target is inside the scan volume and out of the notch."""
        _los, d = line_of_sight(pos_own, pos_tgt)
        if d > self.max_range:
            return False
        if off_boresight(pos_own, act_own, pos_tgt) > self.gimbal_deg:
            return False
        return abs(radial_speed(pos_own, pos_tgt, act_tgt)) >= self.notch_closure

    def in_notch(
        self,
        pos_own: Sequence[float], act_own: Sequence[float],
        pos_tgt: Sequence[float], act_tgt: Sequence[float],
    ) -> bool:
        """Target is beaming: inside the scan volume but Doppler-rejected."""
        _los, d = line_of_sight(pos_own, pos_tgt)
        if d > self.max_range:
            return False
        if off_boresight(pos_own, act_own, pos_tgt) > self.gimbal_deg:
            return False
        return abs(radial_speed(pos_own, pos_tgt, act_tgt)) < self.notch_closure


# --------------------------------------------------------------------- #
# Weapon envelope
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class MissileSpec:
    """Representative active-radar missile (AIM-120C class)."""

    name: str = "ARH"
    rmax: float = 70_000.0          # m, kinematic reach head-on at altitude
    rne: float = 25_000.0           # m, no-escape zone head-on at altitude
    rmin: float = 1_500.0           # m, minimum arming range
    speed: float = 900.0            # m/s average velocity over the flyout
    activation_range: float = 16_000.0   # m to target when the seeker goes active
    max_flight_time: float = 120.0  # s of usable energy
    lethal_radius: float = 120.0    # m


DEFAULT_MISSILE = MissileSpec()


def altitude_factor(alt_m: float) -> float:
    """Thin air buys range: 0.6 at sea level rising to 1.0 by 10 km."""
    return 0.6 + 0.4 * max(0.0, min(1.0, alt_m / 10_000.0))


def aspect_factor(target_aspect_deg: float) -> float:
    """1.0 against a closing target, 0.35 against one running directly away.

    ``target_aspect_deg`` is the angle between the target's velocity and the
    line of sight back to the shooter: 0 = coming at us, 180 = going away.
    """
    a = max(0.0, min(180.0, target_aspect_deg))
    return 1.0 - 0.65 * (a / 180.0)


def kinematic_range(
    spec: MissileSpec, own_alt: float, target_aspect_deg: float,
    base: Optional[float] = None,
) -> float:
    """Reach of the missile in these conditions (Rmax by default, Rne if given)."""
    reach = spec.rmax if base is None else base
    return reach * altitude_factor(own_alt) * aspect_factor(target_aspect_deg)


def launch_authority(
    spec: MissileSpec,
    pos_own: Sequence[float], act_own: Sequence[float],
    pos_tgt: Sequence[float], act_tgt: Sequence[float],
) -> dict:
    """Range bands for a shot right now.

    Returns ``{"range", "rmax", "rne", "rmin", "in_envelope", "in_nez"}``.
    ``in_envelope`` means the shot can reach; ``in_nez`` means the target
    cannot outrun it by turning and running.
    """
    _los, d = line_of_sight(pos_own, pos_tgt)
    feats = geo.situation(pos_own, act_own, pos_tgt, act_tgt)
    # feats[1] is the target's aspect angle: the angle between its velocity and
    # the line of sight back to us.  0 = coming straight at us (longest reach),
    # 180 = running straight away (shortest).  That is exactly what
    # aspect_factor wants, so it goes in unchanged.
    target_aspect = feats[1]
    rmax = kinematic_range(spec, pos_own[2], target_aspect)
    rne = kinematic_range(spec, pos_own[2], target_aspect, base=spec.rne)
    return {
        "range": d,
        "rmax": rmax,
        "rne": rne,
        "rmin": spec.rmin,
        "in_envelope": spec.rmin <= d <= rmax,
        "in_nez": spec.rmin <= d <= rne,
    }


# --------------------------------------------------------------------- #
# Missiles in flight
# --------------------------------------------------------------------- #
@dataclass
class Missile:
    """A guided round in flight.

    Before the seeker goes active the round is flying on the launcher's radar:
    break that lock and it goes stupid.  After it goes active it guides itself,
    but a target that beams it drops into its own Doppler notch, and a round
    that runs out of energy falls out of the sky either way.
    """

    spec: MissileSpec
    shooter: str                      # "agent" or "bandit"
    pos: List[float]
    speed: float
    flight_time: float = 0.0
    active: bool = False              # seeker has gone "pitbull"
    alive: bool = True
    outcome: Optional[str] = None     # "hit", "no_lock", "notched", "out_of_energy"
    _seeker: Radar = field(default_factory=lambda: Radar(gimbal_deg=45.0,
                                                         max_range=20_000.0))
    _heading: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    @classmethod
    def launch(cls, spec: MissileSpec, shooter: str,
               pos: Sequence[float], act: Sequence[float]) -> "Missile":
        vx, vy, vz = geo.velocity_enu(1.0, act[1], act[2])
        return cls(spec=spec, shooter=shooter, pos=list(pos), speed=spec.speed,
                   _heading=(vx, vy, vz))

    # ---------------------------------------------------------------- #
    def step(
        self,
        target_pos: Sequence[float],
        target_act: Sequence[float],
        supported: bool,
        dt: float = 1.0,
    ) -> Optional[str]:
        """Fly one step toward the target.  Returns the outcome if it ends.

        ``supported`` is whether the launching radar still holds the target;
        it only matters before the seeker goes active.
        """
        if not self.alive:
            return self.outcome

        self.flight_time += dt
        los, d = line_of_sight(self.pos, target_pos)

        if not self.active and d <= self.spec.activation_range:
            self.active = True  # pitbull -- the launcher is free to maneuver

        if self.active:
            act = [self.speed, 0.0, 0.0]
            guiding = self._seeker.can_see(self.pos, self._nose_act(), target_pos,
                                           target_act)
            if not guiding:
                return self._end("notched")
        elif not supported:
            return self._end("no_lock")

        if self.flight_time > self.spec.max_flight_time:
            return self._end("out_of_energy")

        # Pure pursuit onto the target: enough to make range, notch and F-pole
        # behave correctly without pretending to model proportional navigation.
        step_len = self.speed * dt
        if d <= max(step_len, self.spec.lethal_radius):
            self.pos = list(target_pos)
            return self._end("hit")
        self._heading = los
        self.pos = [self.pos[i] + los[i] * step_len for i in range(3)]
        return None

    # ---------------------------------------------------------------- #
    def _nose_act(self) -> List[float]:
        """The missile's own (v, gamma, psi) so the seeker geometry works."""
        hx, hy, hz = self._heading
        horiz = math.hypot(hx, hy)
        gamma = math.degrees(math.atan2(hz, max(horiz, 1e-9)))
        psi = geo.wrap_heading(math.degrees(math.atan2(hx, hy)))
        return [self.speed, gamma, psi]

    def _end(self, outcome: str) -> str:
        self.alive = False
        self.outcome = outcome
        return outcome

    def time_to_active(self, target_pos: Sequence[float]) -> float:
        """Seconds of radar support still owed before the seeker takes over.

        This is the number the shooter is actually flying to: until it reaches
        zero, turning away enough to drop the lock throws the shot away.
        """
        if self.active:
            return 0.0
        _los, d = line_of_sight(self.pos, target_pos)
        return max(0.0, (d - self.spec.activation_range) / max(self.speed, 1.0))
