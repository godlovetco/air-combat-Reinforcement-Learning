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
      }
    }

Python -> DCS (default port 7779): a single CSV line so the Lua side needs
no JSON parser::

    pitch,roll,rudder,thrust,trigger\n

with pitch/roll/rudder in [-1, 1], thrust in [0, 1], trigger in {0, 1}.
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


@dataclass
class Telemetry:
    t: float
    own: Ownship
    bandit: Optional[Contact]


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
    bandit = None
    b = data.get("bandit")
    if b:
        bandit = Contact(
            name=b.get("name", "?"),
            pos=(float(b["px"]), float(b["py"]), float(b["pz"])),
            heading=float(b.get("heading", 0.0)),
            pitch=float(b.get("pitch", 0.0)),
        )
    return Telemetry(t=float(data["t"]), own=own, bandit=bandit)


def format_command(c: Controls) -> bytes:
    return (
        f"{c.pitch:.4f},{c.roll:.4f},{c.rudder:.4f},{c.thrust:.4f},{int(c.trigger)}\n"
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
