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
AESA_RANGE = 135_000.0       # m, the same target seen by an AESA
AESA_NOTCH_CLOSURE = 30.0    # m/s; waveform agility narrows the blind zone
# A conventional radar is picked up by a warning receiver well beyond its own
# detection range -- the receiver only has to hear one-way, while the radar
# needs a return trip.  "You are spiked before you are seen" is the normal
# state of affairs.  An LPI array inverts that: it can be holding a track from
# outside the range at which its emissions register at all.
MSA_RWR_FACTOR = 2.0         # of max range at which a conventional set is heard
AESA_LPI_FRACTION = 0.5      # of max range at which an LPI array is heard
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
    """A fighter radar, or a missile seeker.

    Two array types are modeled, and the differences are the ones that change
    how a BVR fight is flown rather than a spec sheet:

    * **Detection range.** An AESA's power management buys roughly half again
      the range of a mechanically-scanned antenna.
    * **Notch width.** A mechanical set filters on one PRF at a time and has a
      wide Doppler blind zone; an AESA hops waveforms and narrows it. Beaming
      still works against both -- it just takes more precision against an AESA.
    * **Re-acquisition.** A mechanical antenna has to physically slew back and
      re-scan after it loses a track; an AESA repositions the beam
      electronically and is back almost immediately. This is why notching a
      mechanical radar buys so much more time.
    * **Simultaneous tracks.** How many missiles the radar can support at once.
    * **LPI.** An AESA can spread its emissions enough that the target's RWR
      does not register the lock until much closer -- you can be shot at
      without knowing it, which the observation block reflects.
    """

    gimbal_deg: float = RADAR_GIMBAL_DEG
    max_range: float = RADAR_RANGE
    notch_closure: float = NOTCH_CLOSURE
    kind: str = "mechanical"
    simultaneous_tracks: int = 1
    reacquire_time: float = 3.0        # s of scan needed to regain a lost track
    lpi_fraction: float = MSA_RWR_FACTOR  # of max_range at which the RWR hears us

    # ---------------------------------------------------------------- #
    @classmethod
    def mechanical(cls, **kw) -> "Radar":
        """Mechanically-scanned array: shorter reach, wide notch, slow to recover."""
        return cls(**kw)

    @classmethod
    def aesa(cls, **kw) -> "Radar":
        """Active electronically scanned array."""
        params = dict(
            gimbal_deg=RADAR_GIMBAL_DEG,
            max_range=AESA_RANGE,
            notch_closure=AESA_NOTCH_CLOSURE,
            kind="aesa",
            simultaneous_tracks=4,
            reacquire_time=0.5,
            lpi_fraction=AESA_LPI_FRACTION,
        )
        params.update(kw)
        return cls(**params)

    def warns_at(self) -> float:
        """Range inside which this radar's emissions trip the target's RWR.

        For a conventional set this is *larger* than its own detection range;
        for an LPI array it is smaller.  That inversion is the whole tactical
        argument for the technology.
        """
        return self.max_range * self.lpi_fraction

    def is_detected_by_rwr(
        self, pos_own: Sequence[float], pos_tgt: Sequence[float]
    ) -> bool:
        """Whether the target's RWR registers a spike from this radar."""
        _los, d = line_of_sight(pos_own, pos_tgt)
        return d <= self.warns_at()

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
    """A weapon's kinematics, guidance and seeker.

    ``guidance`` decides how the round behaves in flight:

    * ``"active_radar"`` -- flies on the launcher's radar until its own seeker
      goes active at ``activation_range``.  Defeatable by breaking the
      supporting lock before then, and by notching the seeker after.
    * ``"infrared"`` -- fire-and-forget from the rail; no radar support, and
      Doppler notching does nothing to it.  It pays for that with range.
    """

    name: str = "ARH"
    guidance: str = "active_radar"
    rmax: float = 70_000.0          # m, kinematic reach head-on at altitude
    rne: float = 25_000.0           # m, no-escape zone head-on at altitude
    rmin: float = 1_500.0           # m, minimum arming range
    speed: float = 900.0            # m/s average velocity over the flyout
    activation_range: float = 16_000.0   # m to target when the seeker goes active
    max_flight_time: float = 120.0  # s of usable energy
    lethal_radius: float = 120.0    # m
    seeker_gimbal_deg: float = 45.0
    seeker_range: float = 20_000.0
    seeker_notch_closure: float = NOTCH_CLOSURE
    seeker_memory: float = 8.0      # s the seeker coasts on its last solution

    def seeker(self) -> Radar:
        return Radar(gimbal_deg=self.seeker_gimbal_deg,
                     max_range=self.seeker_range,
                     notch_closure=self.seeker_notch_closure)

    @property
    def fire_and_forget(self) -> bool:
        return self.guidance == "infrared"


# Representative weapons.  The point of carrying three is that they impose
# different fights: the medium ARH is the workhorse, the ramjet's no-escape
# zone is large enough that "just turn and run" stops working, and the IR
# missile cannot be notched at all but has to be taken into the merge.
ARH_MEDIUM = MissileSpec(
    name="ARH-medium",                 # AIM-120C class
)
ARH_LONG = MissileSpec(
    name="ARH-long",                   # ramjet, Meteor / AIM-120D class
    rmax=120_000.0, rne=60_000.0, rmin=2_000.0, speed=1_000.0,
    activation_range=20_000.0, max_flight_time=200.0,
    seeker_range=25_000.0,
)
IR_SHORT = MissileSpec(
    name="IR-short",                   # AIM-9X class
    guidance="infrared",
    rmax=18_000.0, rne=8_000.0, rmin=300.0, speed=800.0,
    activation_range=float("inf"),     # active off the rail
    max_flight_time=60.0,
    seeker_gimbal_deg=90.0, seeker_range=18_000.0,
    seeker_notch_closure=0.0,          # infrared does not care about Doppler
)

MISSILES = {m.name: m for m in (ARH_MEDIUM, ARH_LONG, IR_SHORT)}
DEFAULT_MISSILE = ARH_MEDIUM


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


@dataclass(frozen=True)
class WeaponEngagementZone:
    """The range bands for a shot taken right now.

    ``rmax``  the round can reach the target if it keeps doing what it is doing.
    ``rtr``   "turn and run": inside this, a target that reverses still dies.
    ``rne``   no-escape zone -- no maneuver defeats it kinematically.
    ``rmin``  minimum arming range.
    """

    range: float
    rmax: float
    rtr: float
    rne: float
    rmin: float
    time_of_flight: float

    @property
    def in_envelope(self) -> bool:
        return self.rmin <= self.range <= self.rmax

    @property
    def in_nez(self) -> bool:
        return self.rmin <= self.range <= self.rne

    @property
    def defeatable_by_running(self) -> bool:
        """A shot that reaches now but that the target can still outrun."""
        return self.in_envelope and self.range > self.rtr

    def as_dict(self) -> dict:
        return {
            "range": self.range, "rmax": self.rmax, "rtr": self.rtr,
            "rne": self.rne, "rmin": self.rmin,
            "time_of_flight": self.time_of_flight,
            "in_envelope": self.in_envelope, "in_nez": self.in_nez,
        }


def weapon_engagement_zone(
    spec: MissileSpec,
    pos_own: Sequence[float], act_own: Sequence[float],
    pos_tgt: Sequence[float], act_tgt: Sequence[float],
) -> WeaponEngagementZone:
    """WEZ against this target in this geometry."""
    _los, d = line_of_sight(pos_own, pos_tgt)
    feats = geo.situation(pos_own, act_own, pos_tgt, act_tgt)
    # feats[1] is the target's aspect angle: the angle between its velocity and
    # the line of sight back to us.  0 = coming straight at us (longest reach),
    # 180 = running straight away (shortest).
    target_aspect = feats[1]
    rmax = kinematic_range(spec, pos_own[2], target_aspect)
    rne = kinematic_range(spec, pos_own[2], target_aspect, base=spec.rne)
    # Rtr is what the shot is worth if the target reverses the moment it
    # launches: recompute the reach against a target running directly away.
    rtr = kinematic_range(spec, pos_own[2], 180.0)
    closing = max(1.0, spec.speed + closure_rate(pos_own, act_own, pos_tgt, act_tgt))
    return WeaponEngagementZone(range=d, rmax=rmax, rtr=rtr, rne=rne,
                                rmin=spec.rmin, time_of_flight=d / closing)


def launch_authority(
    spec: MissileSpec,
    pos_own: Sequence[float], act_own: Sequence[float],
    pos_tgt: Sequence[float], act_tgt: Sequence[float],
) -> dict:
    """``weapon_engagement_zone`` as a plain dict, for callers that want one."""
    return weapon_engagement_zone(spec, pos_own, act_own, pos_tgt, act_tgt).as_dict()


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
    memory_left: Optional[float] = None   # s of inertial coast still available
    _seeker: Optional[Radar] = None
    _heading: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self):
        if self._seeker is None:
            self._seeker = self.spec.seeker()

    @classmethod
    def launch(cls, spec: MissileSpec, shooter: str,
               pos: Sequence[float], act: Sequence[float]) -> "Missile":
        vx, vy, vz = geo.velocity_enu(1.0, act[1], act[2])
        return cls(spec=spec, shooter=shooter, pos=list(pos), speed=spec.speed,
                   active=spec.fire_and_forget,   # IR guides off the rail
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

        guiding = True
        if self.active:
            guiding = self._seeker.can_see(self.pos, self._nose_act(), target_pos,
                                           target_act)
            if guiding:
                self.memory_left = None       # solution is good again
            else:
                # Notched, but not dead.  The seeker coasts on its last
                # solution: a target has to *hold* the beam long enough for the
                # missile to fly past or run out of energy.  Come out of the
                # notch early and it reacquires.  Killing the round the instant
                # it lost the return made a single perfect beam turn defeat
                # every shot, which turned every engagement into a merge.
                if self.memory_left is None:
                    self.memory_left = self.spec.seeker_memory
                self.memory_left -= dt
                if self.memory_left <= 0.0:
                    return self._end("notched")
        elif not supported:
            return self._end("no_lock")

        if self.flight_time > self.spec.max_flight_time:
            return self._end("out_of_energy")

        # Pure pursuit onto the target while the solution is good; straight
        # ahead on the last known heading while coasting.  Enough to make
        # range, notch and F-pole behave correctly without pretending to model
        # proportional navigation.
        step_len = self.speed * dt
        if guiding:
            if d <= max(step_len, self.spec.lethal_radius):
                self.pos = list(target_pos)
                return self._end("hit")
            self._heading = los
        elif d <= self.spec.lethal_radius:
            return self._end("hit")
        self.pos = [self.pos[i] + self._heading[i] * step_len for i in range(3)]
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

    def is_tracking(
        self, target_pos: Sequence[float], target_act: Sequence[float]
    ) -> bool:
        """Whether an active round's own seeker currently holds the target.

        False for a round still on datalink -- it is not tracking anything
        itself yet -- and false for one whose target has beamed into the notch.
        """
        if not (self.alive and self.active):
            return False
        return self._seeker.can_see(self.pos, self._nose_act(), target_pos, target_act)

    @property
    def coasting(self) -> bool:
        """Active, but flying on memory rather than on a live return."""
        return self.alive and self.active and self.memory_left is not None

    def time_to_active(self, target_pos: Sequence[float]) -> float:
        """Seconds of radar support still owed before the seeker takes over.

        This is the number the shooter is actually flying to: until it reaches
        zero, turning away enough to drop the lock throws the shot away.
        """
        if self.active:
            return 0.0
        _los, d = line_of_sight(self.pos, target_pos)
        return max(0.0, (d - self.spec.activation_range) / max(self.speed, 1.0))
