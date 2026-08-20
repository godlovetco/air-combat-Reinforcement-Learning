"""BVR decisions from live DCS telemetry.

:mod:`dcs_bridge.bvr_env` runs the missile fight in simulation, where the
environment knows everything.  In DCS it does not: the export gives a radar
lock, an RWR emitter list and a stores count, and nothing at all about the
rounds already in the air.  This module is the difference between those two
worlds, kept apart from the flight loop so it can be tested without a socket.

Two things here are dead-reckoned rather than measured, and both are called out
where they are used:

* **Our own shot's timeline.**  DCS does not export the state of a missile we
  fired, so the seconds of radar support still owed are estimated from the
  launch range and the weapon's average velocity.  It is an estimate; it drives
  when the pilot stops cranking and is deliberately conservative.
* **The threat's time to impact.**  A warning receiver reports bearing and a
  launch flag, not range.  When it cannot be inferred the field is ``None``,
  and every consumer treats "unknown" as "defend now" rather than guessing a
  comfortable number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from . import bvr
from . import geometry as geo
from .bvr_pilot import TimelinePilot

MIN_SHOT_INTERVAL = 8.0     # s of trigger discipline between releases
SUPPORT_MARGIN = 2.0        # s of slop kept on the support estimate


def _bearing(from_pos, to_pos) -> float:
    return geo.wrap_heading(math.degrees(math.atan2(
        to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])))


@dataclass
class Shot:
    """A round we released, tracked by dead reckoning.

    DCS tells us nothing about it after the rail, so its position is inferred
    from the launch range and the weapon's average velocity.
    """

    spec: bvr.MissileSpec
    launched_at: float
    launch_range: float

    def flight_time(self, t: float) -> float:
        return max(0.0, t - self.launched_at)

    def range_to_target(self, t: float, target_range: float) -> float:
        """Estimated missile-to-target distance.

        The round is assumed to fly the line of sight at its average velocity,
        so what it has closed is its own travel plus whatever the target has
        closed on the shooter since launch.
        """
        travelled = self.spec.speed * self.flight_time(t)
        closed = max(0.0, self.launch_range - target_range)
        return max(0.0, self.launch_range - travelled - closed)

    def support_owed(self, t: float, target_range: float) -> float:
        """Seconds of radar support left before the seeker takes over."""
        gap = self.range_to_target(t, target_range) - self.spec.activation_range
        if gap <= 0.0:
            return 0.0
        return gap / max(self.spec.speed, 1.0) + SUPPORT_MARGIN

    def active(self, t: float, target_range: float) -> bool:
        return self.support_owed(t, target_range) <= 0.0

    def expired(self, t: float) -> bool:
        return self.flight_time(t) > self.spec.max_flight_time


@dataclass
class BVRDecision:
    """What the controller wants done this tick."""

    situation: dict
    desired_heading: Optional[float] = None
    weapon: int = 0
    reason: str = ""


class BVRController:
    """Live BVR brain: situation block, maneuver, and the shot decision."""

    def __init__(
        self,
        spec: bvr.MissileSpec = bvr.DEFAULT_MISSILE,
        pilot: Optional[TimelinePilot] = None,
        min_shot_interval: float = MIN_SHOT_INTERVAL,
        radar: Optional[bvr.Radar] = None,
    ):
        self.spec = spec
        self.pilot = pilot or TimelinePilot()
        self.min_shot_interval = min_shot_interval
        self.radar = radar or bvr.Radar.mechanical()
        self.shots: List[Shot] = []
        self._last_shot_t = -1e9

    # ------------------------------------------------------------------ #
    def _prune(self, t: float, target_range: float) -> None:
        self.shots = [s for s in self.shots if not s.expired(t)]

    def _threat(self, telem) -> Optional[dict]:
        sensors = getattr(telem, "sensors", None)
        if sensors is None:
            return None
        worst = sensors.nearest_threat()
        if worst is None or not (worst.launch or worst.lock):
            return None
        return {
            # An RWR gives bearing, not range: seconds is unknown and stays
            # unknown rather than being invented.
            "off_nose_deg": geo.heading_error(
                geo.wrap_heading(telem.own.heading + worst.az), telem.own.heading),
            "seconds": None,
            "active": bool(worst.launch),
            "notch_depth": math.cos(math.radians(worst.az)),
            "from_rwr": True,
        }

    # ------------------------------------------------------------------ #
    def situation(self, telem) -> dict:
        """The block :class:`TimelinePilot` and the WSO advisor both read."""
        own = telem.own
        sensors = getattr(telem, "sensors", None)
        bandit = telem.bandit

        target_range = math.dist(own.pos, bandit.pos) if bandit is not None else 0.0
        self._prune(telem.t, target_range)

        wez = None
        if bandit is not None:
            act_own = [own.tas, own_gamma_from(own), own.heading]
            act_tgt = [own.tas, bandit.pitch, bandit.heading]  # speed unknown
            wez = bvr.weapon_engagement_zone(
                self.spec, list(own.pos), act_own, list(bandit.pos), act_tgt)

        supported = [s for s in self.shots if not s.active(telem.t, target_range)]
        owed = (max(s.support_owed(telem.t, target_range) for s in supported)
                if supported else 0.0)

        block = {
            "locked": bool(sensors.locked) if sensors else False,
            "spiked": bool(sensors.spiked) if sensors else False,
            "threat": self._threat(telem),
            "missiles": (sensors.missiles if sensors and sensors.missiles is not None
                         else 0),
            "shot_in_flight": bool(self.shots),
            "support_owed_s": owed,
            "wez": ({**wez.as_dict(), "threatened": False} if wez else {}),
        }
        return {
            "own": {"heading_deg": own.heading, "gamma_deg": own_gamma_from(own),
                    "tas": own.tas, "altitude_m": own.pos[2]},
            "target_bearing_deg": (_bearing(own.pos, bandit.pos)
                                   if bandit is not None else own.heading),
            "bvr": block,
        }

    # ------------------------------------------------------------------ #
    def decide(self, telem, weapons_free: bool = False) -> BVRDecision:
        sit = self.situation(telem)
        block = sit["bvr"]
        heading = self.pilot.desired_heading(sit)

        if not weapons_free:
            return BVRDecision(sit, heading, 0, "weapons hold")
        if telem.bandit is None:
            return BVRDecision(sit, heading, 0, "no target")
        if not block["locked"]:
            return BVRDecision(sit, heading, 0, "no lock")
        if block["missiles"] <= 0:
            return BVRDecision(sit, heading, 0, "winchester")
        if telem.t - self._last_shot_t < self.min_shot_interval:
            return BVRDecision(sit, heading, 0, "trigger discipline")
        # One round in the air at a time unless the radar can guide more.  In a
        # 1v1 this is usually subsumed by the shot-quality rule below -- a
        # second shot far enough out to still need support is also far enough
        # out to be defeatable -- so it rarely decides anything on its own.  It
        # is kept because it is the physically real constraint and it is what
        # will bind once there is more than one target to shoot at.
        unsupported = [s for s in self.shots
                       if not s.active(telem.t,
                                       math.dist(telem.own.pos, telem.bandit.pos))]
        if len(unsupported) >= self.radar.simultaneous_tracks:
            return BVRDecision(sit, heading, 0, "already supporting a shot")
        wez = block["wez"]
        if not wez.get("in_envelope"):
            return BVRDecision(sit, heading, 0, "out of the envelope")
        if not wez.get("in_nez") and unsupported:
            return BVRDecision(sit, heading, 0, "defeatable shot, one already up")
        return BVRDecision(sit, heading, 1,
                           "in the no-escape zone" if wez.get("in_nez")
                           else "in the envelope")

    def confirm_shot(self, telem) -> None:
        """Record a release.  Call this only when the command actually went out."""
        rng = (math.dist(telem.own.pos, telem.bandit.pos)
               if telem.bandit is not None else 0.0)
        self.shots.append(Shot(self.spec, telem.t, rng))
        self._last_shot_t = telem.t


def own_gamma_from(own) -> float:
    """Flight-path angle from true airspeed and vertical velocity."""
    if own.tas <= 1.0:
        return 0.0
    return math.degrees(math.asin(max(-1.0, min(1.0, own.vv / own.tas))))
