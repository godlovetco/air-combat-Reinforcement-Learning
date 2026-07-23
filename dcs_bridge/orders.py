"""Tactical orders and shared pilot state for the radio wingman.

The LLM radio agent (``wingman.py``) translates flight-lead transmissions
into :class:`TacticalOrder` objects; the flight loop (``run_pilot.py``)
consumes them through a thread-safe :class:`PilotState`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

ORDER_TYPES = (
    "engage",        # commit on the nearest hostile (RL policy / lead pursuit)
    "anchor",        # hold position: orbit at current location and altitude
    "vector",        # fly a commanded heading (optionally altitude/speed)
    "break_left",    # immediate hard left turn (defensive)
    "break_right",   # immediate hard right turn (defensive)
    "rtb",           # return to the position where the flight started
    "weapons_free",  # allow trigger inside the gun envelope
    "weapons_hold",  # forbid weapons release
    "status",        # no maneuver change; radio a status report
    # CCA / manned-unmanned teaming (MUM-T)
    "formation",     # rejoin and hold a named formation on the crewed lead
    "rejoin",        # return from engagement to formation on the lead
    "scout",         # run ahead of the lead to sweep for contacts
    "leash",         # set supervised-autonomy level (close/tight/loose)
)

# Supervised-autonomy "leash" levels, from most to least restrictive.
#   close  -- station-keeping only; never leaves formation on its own
#   tight  -- may commit on threats but weapons stay caged until cleared
#   loose  -- may commit and fire within the gun envelope without a call
LEASH_LEVELS = ("close", "tight", "loose")
DEFAULT_LEASH = "tight"


@dataclass
class TacticalOrder:
    order: str
    heading_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    speed_ms: Optional[float] = None
    station: Optional[str] = None  # formation name for "formation" orders
    side: Optional[str] = None     # "left" / "right" station side
    leash: Optional[str] = None    # supervised-autonomy level for "leash" orders

    def __post_init__(self) -> None:
        if self.order not in ORDER_TYPES:
            raise ValueError(f"unknown order {self.order!r}, expected one of {ORDER_TYPES}")
        if self.side is not None and self.side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")
        if self.leash is not None and self.leash not in LEASH_LEVELS:
            raise ValueError(f"unknown leash {self.leash!r}, expected one of {LEASH_LEVELS}")

    @classmethod
    def from_tool_input(cls, data: dict) -> "TacticalOrder":
        return cls(
            order=data["order"],
            heading_deg=data.get("heading_deg"),
            altitude_m=data.get("altitude_m"),
            speed_ms=data.get("speed_ms"),
            station=data.get("station"),
            side=data.get("side"),
            leash=data.get("leash"),
        )

    def describe(self) -> str:
        parts = [self.order.replace("_", " ")]
        if self.station is not None:
            parts.append(self.station.replace("_", " "))
        if self.side is not None:
            parts.append(f"{self.side} side")
        if self.leash is not None:
            parts.append(f"leash {self.leash}")
        if self.heading_deg is not None:
            parts.append(f"heading {self.heading_deg:03.0f}")
        if self.altitude_m is not None:
            parts.append(f"altitude {self.altitude_m:.0f} m")
        if self.speed_ms is not None:
            parts.append(f"speed {self.speed_ms:.0f} m/s")
        return ", ".join(parts)


class PilotState:
    """Mutable flight-loop state shared between the radio thread and the
    20 Hz control loop.  All access goes through the lock."""

    def __init__(
        self,
        weapons_free: bool = False,
        target_speed: float = 250.0,
        mode: str = "engage",
        formation_station: str = "combat_spread",
        formation_side: str = "right",
        leash: str = DEFAULT_LEASH,
    ):
        self._lock = threading.Lock()
        self.mode = mode
        self.weapons_free = weapons_free
        self.vector_heading: Optional[float] = None
        self.vector_altitude: Optional[float] = None
        self.target_speed = target_speed
        self.pending_break: Optional[str] = None  # "left" / "right"
        self._home: Optional[tuple] = None        # ENU position captured at start
        # CCA / MUM-T state
        self.formation_station = formation_station
        self.formation_side = formation_side
        self.leash = leash
        # mode to return to when an auto-commit ends; None when the lead
        # commanded the engagement explicitly (we do not auto-rejoin then).
        self._rejoin_mode: Optional[str] = None

    def apply(self, order: TacticalOrder) -> None:
        with self._lock:
            if order.order in ("engage", "anchor"):
                self.mode = order.order
                self.pending_break = None
            elif order.order == "vector":
                self.mode = "vector"
                self.pending_break = None
                if order.heading_deg is not None:
                    self.vector_heading = order.heading_deg % 360.0
                if order.altitude_m is not None:
                    self.vector_altitude = order.altitude_m
            elif order.order in ("break_left", "break_right"):
                # Resolved into a vector by the flight loop, which knows the
                # current heading.
                self.pending_break = order.order.split("_")[1]
            elif order.order == "rtb":
                self.mode = "rtb"
                self.pending_break = None
            elif order.order in ("formation", "rejoin"):
                self.mode = "formation"
                self.pending_break = None
                self._rejoin_mode = None
                if order.station is not None:
                    self.formation_station = order.station
                if order.side is not None:
                    self.formation_side = order.side
            elif order.order == "scout":
                self.mode = "scout"
                self.pending_break = None
                self._rejoin_mode = None
            elif order.order == "leash":
                if order.leash is not None:
                    self.leash = order.leash
                    # close leash cages weapons and pins the CCA to formation;
                    # loose leash grants standing weapons-free.
                    if order.leash == "close":
                        self.weapons_free = False
                        if self.mode in ("engage", "scout"):
                            self.mode = "formation"
                            self._rejoin_mode = None
                    elif order.leash == "loose":
                        self.weapons_free = True
            elif order.order == "weapons_free":
                self.weapons_free = True
            elif order.order == "weapons_hold":
                self.weapons_free = False
            # "status" changes nothing
            if order.speed_ms is not None:
                self.target_speed = order.speed_ms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "weapons_free": self.weapons_free,
                "vector_heading": self.vector_heading,
                "vector_altitude": self.vector_altitude,
                "target_speed": self.target_speed,
                "pending_break": self.pending_break,
                "formation_station": self.formation_station,
                "formation_side": self.formation_side,
                "leash": self.leash,
            }

    def auto_commit(self, threat_in_range: bool) -> bool:
        """Supervised-autonomy commit: leave formation/scout to engage a threat.

        Only fires when the leash allows autonomous commit (tight or loose)
        and the CCA is currently holding formation or scouting.  Remembers the
        prior mode so :meth:`auto_rejoin` can send it back.  Returns True on the
        transition so the flight loop can make a one-time radio call.
        """
        with self._lock:
            if not threat_in_range or self.leash == "close":
                return False
            if self.mode not in ("formation", "scout"):
                return False
            self._rejoin_mode = self.mode
            self.mode = "engage"
            return True

    def auto_rejoin(self, threat_gone: bool) -> bool:
        """End an autonomous engagement and return to the prior formation mode.

        No-op when the engagement was commanded by the lead (``_rejoin_mode``
        is None) so a lead-ordered ``engage`` is never silently abandoned.
        Returns True on the transition.
        """
        with self._lock:
            if not threat_gone or self.mode != "engage" or self._rejoin_mode is None:
                return False
            self.mode = self._rejoin_mode
            self._rejoin_mode = None
            return True

    def resolve_break(self, current_heading: float) -> None:
        """Turn a pending break order into a hard 90-degree vector."""
        with self._lock:
            if self.pending_break is None:
                return
            sign = -1.0 if self.pending_break == "left" else 1.0
            self.vector_heading = (current_heading + sign * 90.0) % 360.0
            self.vector_altitude = None
            self.mode = "vector"
            self.pending_break = None

    def set_home(self, pos) -> None:
        with self._lock:
            if self._home is None:
                self._home = (pos[0], pos[1], pos[2])

    @property
    def home(self) -> Optional[tuple]:
        with self._lock:
            return self._home

    def apply_anchor_arrival(self) -> None:
        """RTB complete: switch to holding overhead the home point."""
        with self._lock:
            if self.mode == "rtb":
                self.mode = "anchor"
