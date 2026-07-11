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
)


@dataclass
class TacticalOrder:
    order: str
    heading_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    speed_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.order not in ORDER_TYPES:
            raise ValueError(f"unknown order {self.order!r}, expected one of {ORDER_TYPES}")

    @classmethod
    def from_tool_input(cls, data: dict) -> "TacticalOrder":
        return cls(
            order=data["order"],
            heading_deg=data.get("heading_deg"),
            altitude_m=data.get("altitude_m"),
            speed_ms=data.get("speed_ms"),
        )

    def describe(self) -> str:
        parts = [self.order.replace("_", " ")]
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

    def __init__(self, weapons_free: bool = False, target_speed: float = 250.0):
        self._lock = threading.Lock()
        self.mode = "engage"
        self.weapons_free = weapons_free
        self.vector_heading: Optional[float] = None
        self.vector_altitude: Optional[float] = None
        self.target_speed = target_speed
        self.pending_break: Optional[str] = None  # "left" / "right"
        self._home: Optional[tuple] = None        # ENU position captured at start

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
            }

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
