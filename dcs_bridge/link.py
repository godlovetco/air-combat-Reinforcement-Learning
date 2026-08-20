"""UDP link between DCS World (Lua export script) and the Python AI pilot.

Protocol
--------
DCS -> Python (default port 7778): one JSON object per datagram, produced by
``dcs-addon/Scripts/UCAVPilot/UCAVPilotExport.lua``::

    {
      "t": 1234.56,                      -- DCS model time, seconds
      "own": {
        "name": "...", "px": E, "py": N, "pz": U,   -- ENU meters
        "heading": deg, "pitch": deg, "bank": deg,
        "tas": m/s, "vv": m/s
      },
      "bandit": {                        -- omitted when no hostile airborne
        "name": "...", "px": E, "py": N, "pz": U, "heading": deg, "pitch": deg
      },
      "lead": {                          -- omitted when no friendly lead airborne
        "name": "...", "px": E, "py": N, "pz": U, "heading": deg, "pitch": deg,
        "tas": m/s                       -- crewed flight lead the CCA teams with
      },
      "sensors": {                       -- omitted by older exports
        "locked": true,                  -- own radar holds a track
        "lock_range": m, "lock_az": deg, "lock_el": deg,
        "missiles": 4,                   -- air-to-air rounds remaining
        "threats": [                     -- RWR emitters
          {"az": deg, "power": 0..1, "launch": true, "lock": true}
        ]
      }
    }

Everything under ``sensors`` is optional and every consumer treats its absence
as "no BVR picture", so a mission flown with an older copy of the Lua script --
or an airframe whose module exports no radar page -- degrades to the gun fight
rather than failing.

Python -> DCS (default port 7779): a single CSV line so the Lua side needs
no JSON parser::

    pitch,roll,rudder,thrust,trigger,weapon\n

with pitch/roll/rudder in [-1, 1], thrust in [0, 1], and trigger/weapon in
{0, 1}.  ``trigger`` is the gun, held down while set; ``weapon`` is a missile
release, edge-triggered on the Lua side so one command sends one round.  Lua
scripts that parse only five fields ignore the sixth.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Optional, Tuple

from .autopilot import Controls

DEFAULT_TELEMETRY_PORT = 7778
DEFAULT_COMMAND_PORT = 7779


@dataclass
class Ownship:
    name: str
    pos: Tuple[float, float, float]  # ENU meters (east, north, up)
    heading: float                   # deg
    pitch: float                     # deg
    bank: float                      # deg
    tas: float                       # m/s
    vv: float                        # m/s vertical velocity


@dataclass
class Contact:
    name: str
    pos: Tuple[float, float, float]
    heading: float
    pitch: float
    tas: Optional[float] = None  # m/s, known for the friendly lead (datalink)


@dataclass
class RwrContact:
    """One emitter on the radar warning receiver."""

    az: float             # deg off the nose, signed
    power: float = 0.0    # 0..1, the RWR's own strength indication
    launch: bool = False  # a launch warning is associated with this emitter
    lock: bool = False    # the emitter is in a tracking, not a search, mode


@dataclass
class Sensors:
    """Radar, RWR and stores state.  Every field is optional."""

    locked: bool = False
    lock_range: Optional[float] = None   # m
    lock_az: Optional[float] = None      # deg off the nose
    lock_el: Optional[float] = None      # deg
    missiles: Optional[int] = None       # air-to-air rounds remaining
    threats: Tuple[RwrContact, ...] = ()

    @property
    def spiked(self) -> bool:
        return any(t.lock or t.launch for t in self.threats)

    @property
    def launch_warning(self) -> bool:
        return any(t.launch for t in self.threats)

    def nearest_threat(self) -> Optional[RwrContact]:
        """The emitter to defend against: a launch first, then the strongest."""
        if not self.threats:
            return None
        return max(self.threats, key=lambda t: (t.launch, t.lock, t.power))


@dataclass
class Telemetry:
    t: float
    own: Ownship
    bandit: Optional[Contact]
    lead: Optional[Contact] = None  # crewed flight lead for MUM-T / CCA teaming
    sensors: Optional[Sensors] = None  # radar/RWR/stores; None on older exports


def parse_telemetry(payload: bytes) -> Telemetry:
    data = json.loads(payload.decode("utf-8", errors="replace"))
    o = data["own"]
    own = Ownship(
        name=o.get("name", "?"),
        pos=(float(o["px"]), float(o["py"]), float(o["pz"])),
        heading=float(o["heading"]),
        pitch=float(o["pitch"]),
        bank=float(o["bank"]),
        tas=float(o["tas"]),
        vv=float(o["vv"]),
    )
    return Telemetry(
        t=float(data["t"]),
        own=own,
        bandit=_parse_contact(data.get("bandit")),
        lead=_parse_contact(data.get("lead")),
        sensors=_parse_sensors(data.get("sensors")),
    )


def _parse_contact(c: Optional[dict]) -> Optional[Contact]:
    if not c:
        return None
    tas = c.get("tas")
    return Contact(
        name=c.get("name", "?"),
        pos=(float(c["px"]), float(c["py"]), float(c["pz"])),
        heading=float(c.get("heading", 0.0)),
        pitch=float(c.get("pitch", 0.0)),
        tas=None if tas is None else float(tas),
    )


def _opt_float(v) -> Optional[float]:
    return None if v is None else float(v)


def _parse_sensors(sn: Optional[dict]) -> Optional[Sensors]:
    if not sn:
        return None
    threats = tuple(
        RwrContact(
            az=float(t.get("az", 0.0)),
            power=float(t.get("power", 0.0)),
            launch=bool(t.get("launch", False)),
            lock=bool(t.get("lock", False)),
        )
        for t in (sn.get("threats") or ())
    )
    missiles = sn.get("missiles")
    return Sensors(
        locked=bool(sn.get("locked", False)),
        lock_range=_opt_float(sn.get("lock_range")),
        lock_az=_opt_float(sn.get("lock_az")),
        lock_el=_opt_float(sn.get("lock_el")),
        missiles=None if missiles is None else int(missiles),
        threats=threats,
    )


def format_command(c: Controls) -> bytes:
    return (
        f"{c.pitch:.4f},{c.roll:.4f},{c.rudder:.4f},{c.thrust:.4f},"
        f"{int(c.trigger)},{int(getattr(c, 'weapon', 0))}\n"
    ).encode("ascii")


class DCSLink:
    """Receive telemetry from and send commands to the Lua export script."""

    def __init__(
        self,
        listen_host: str = "127.0.0.1",
        telemetry_port: int = DEFAULT_TELEMETRY_PORT,
        dcs_host: str = "127.0.0.1",
        command_port: int = DEFAULT_COMMAND_PORT,
        timeout: float = 1.0,
    ):
        self.timeout = timeout
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((listen_host, telemetry_port))
        self.rx.settimeout(timeout)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dcs_addr = (dcs_host, command_port)

    def receive(self) -> Optional[Telemetry]:
        """Latest telemetry packet, or None on timeout / bad packet.

        Drains the socket so a slow consumer never falls behind DCS.
        """
        try:
            payload, _ = self.rx.recvfrom(65536)
        except socket.timeout:
            return None
        # Drain any queued older packets, keep only the newest.
        self.rx.setblocking(False)
        try:
            while True:
                try:
                    payload, _ = self.rx.recvfrom(65536)
                except (BlockingIOError, InterruptedError):
                    break
        finally:
            self.rx.settimeout(self.timeout)
        try:
            return parse_telemetry(payload)
        except (ValueError, KeyError):
            return None

    def send(self, controls: Controls) -> None:
        self.tx.sendto(format_command(controls), self.dcs_addr)

    def close(self) -> None:
        self.rx.close()
        self.tx.close()
