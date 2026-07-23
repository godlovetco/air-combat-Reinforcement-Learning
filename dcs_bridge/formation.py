"""CCA loyal-wingman formation station-keeping.

Motivated by the U.S. Air Force **Collaborative Combat Aircraft (CCA)**
program: uncrewed "loyal wingman" aircraft that team with a crewed fighter,
holding tactical formation and executing the flight lead's intent under
supervised autonomy.  This module is the manned-unmanned-teaming (MUM-T)
core -- given the crewed lead's position, heading and speed, it computes the
world point the CCA should occupy for a named formation and a speed law that
paces the lead while closing any station error.

A formation "station" is expressed relative to the lead's nose:

    rel_bearing  deg   clockwise from the lead's heading (0 = ahead,
                       90 = right wing, 180 = astern, 270 = left wing)
    range        m     distance from the lead to the station
    stack        m     altitude offset (positive = CCA above the lead)

The named formations below are the standard two-ship geometries a CCA would
fly; ``side`` mirrors a right-hand station to the left.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from .geometry import wrap_heading

Vec3 = Tuple[float, float, float]

DEG = math.pi / 180.0

# name -> (right-side relative bearing deg, range m, stack m)
FORMATIONS = {
    "line_abreast":   (90.0,  1_800.0,   0.0),   # abeam, co-altitude
    "combat_spread":  (95.0,  1_400.0, 150.0),   # tactical: abeam, slightly high
    "wall":           (90.0,  3_000.0,   0.0),   # wide line abreast
    "fighting_wing":  (125.0,   300.0,   0.0),   # close, ~45 deg back (admin/route)
    "echelon":        (135.0, 1_000.0,   0.0),   # 45 deg back
    "wedge":          (120.0, 1_200.0, 100.0),   # CCA default: back-and-stacked
    "trail":          (180.0, 1_500.0,   0.0),   # directly astern
}

DEFAULT_FORMATION = "combat_spread"
SCOUT_AHEAD_M = 5_000.0  # how far ahead of the lead a scouting CCA flies


def station_offset(name: str, side: str = "right") -> Tuple[float, float, float]:
    """(rel_bearing_deg, range_m, stack_m) for a named formation and side."""
    if name not in FORMATIONS:
        raise ValueError(f"unknown formation {name!r}, expected one of {sorted(FORMATIONS)}")
    rel_bearing, rng, stack = FORMATIONS[name]
    if side == "left":
        rel_bearing = (360.0 - rel_bearing) % 360.0
    elif side != "right":
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return rel_bearing, rng, stack


def station_position(
    lead_pos: Sequence[float],
    lead_heading: float,
    name: str = DEFAULT_FORMATION,
    side: str = "right",
) -> Vec3:
    """World ENU point the CCA should occupy to hold ``name`` on the lead."""
    rel_bearing, rng, stack = station_offset(name, side)
    absolute = wrap_heading(lead_heading + rel_bearing) * DEG
    # ENU: x east, y north; compass bearing measured clockwise from north.
    return (
        lead_pos[0] + rng * math.sin(absolute),
        lead_pos[1] + rng * math.cos(absolute),
        lead_pos[2] + stack,
    )


def scout_position(lead_pos: Sequence[float], lead_heading: float,
                   ahead_m: float = SCOUT_AHEAD_M) -> Vec3:
    """A point ``ahead_m`` in front of the lead along its heading (scout run)."""
    h = wrap_heading(lead_heading) * DEG
    return (
        lead_pos[0] + ahead_m * math.sin(h),
        lead_pos[1] + ahead_m * math.cos(h),
        lead_pos[2],
    )


def station_error(own_pos: Sequence[float], station: Sequence[float]) -> float:
    """Slant distance from the CCA to its station, in meters."""
    return math.dist(own_pos, station)


def in_position(own_pos: Sequence[float], station: Sequence[float],
                tolerance_m: float = 250.0) -> bool:
    """True when the CCA is holding station within ``tolerance_m``."""
    return station_error(own_pos, station) <= tolerance_m


def formation_speed(
    lead_speed: float,
    own_pos: Sequence[float],
    station: Sequence[float],
    lead_heading: float,
    max_delta: float = 80.0,
    gain: float = 0.15,
) -> float:
    """Commanded TAS to pace the lead and close the station error.

    Base speed matches the lead; the along-track component of the station
    error adds/subtracts up to ``max_delta`` m/s so the CCA catches up when
    it lags and eases off when it overshoots.  A never-below floor keeps a
    jet from commanding an unflyably low speed while repositioning.
    """
    h = wrap_heading(lead_heading) * DEG
    ahead = (math.sin(h), math.cos(h))  # unit vector along lead heading (E, N)
    along = (station[0] - own_pos[0]) * ahead[0] + (station[1] - own_pos[1]) * ahead[1]
    delta = max(-max_delta, min(max_delta, gain * along))
    return max(120.0, lead_speed + delta)
