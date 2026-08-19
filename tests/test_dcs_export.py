"""Run the DCS Lua export against a stubbed DCS API.

DCS cannot be run here, and the Lua <-> Python boundary is the highest-risk
interface in the product: a typo in a format string ships a packet the bridge
cannot parse, and nothing in the Python test suite would notice.  So the actual
export script is loaded into a Lua interpreter with the DCS globals stubbed
out, driven for a frame, and the datagram it produces is parsed by the same
``parse_telemetry`` the bridge uses.

Skipped when ``lupa`` is not installed; it is a development dependency, not a
runtime one.
"""

import json
import os
import unittest

try:
    import lupa
except ImportError:  # pragma: no cover - exercised by the skip
    lupa = None

from dcs_bridge.autopilot import Controls
from dcs_bridge.link import format_command, parse_telemetry

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "dcs-addon", "Scripts", "UCAVPilot", "UCAVPilotExport.lua")

# A minimal DCS: own-ship at 8 km, one hostile 40 km north, one friendly lead,
# a radar lock, four missiles and two RWR emitters, one of them shooting.
STUB = r"""
local sent, received = {}, {}
local api = {}

function api.packets() return sent end
function api.push(line) received[#received + 1] = line end

lfs = { writedir = function() return "/tmp/" end }

local function fakeSocket()
    return {
        setpeername = function() return 1 end,
        setsockname = function() return 1 end,
        settimeout  = function() return 1 end,
        sendto      = function(_, packet) sent[#sent + 1] = packet end,
        receive     = function() return table.remove(received, 1) end,
        close       = function() end,
    }
end
package = package or {}
package.loaded = package.loaded or {}
package.loaded.socket = { udp = fakeSocket }

LoGetModelTime        = function() return 12.5 end
LoGetTrueAirSpeed     = function() return 250.0 end
LoGetVerticalVelocity = function() return -3.0 end
LoGetPlayerPlaneId    = function() return 1 end
LoGetSelfData = function()
    return { Name = "F-16C_50",
             Position = { x = 100000, y = 8000, z = 50000 },
             Heading = 0.0, Pitch = 0.05, Bank = -0.1 }
end
LoGetWorldObjects = function()
    return {
        [1] = { Name = "self", CoalitionID = 2, Type = { level1 = 1 },
                Position = { x = 100000, y = 8000, z = 50000 } },
        [2] = { Name = "MiG-29S", CoalitionID = 1, Type = { level1 = 1 },
                Position = { x = 140000, y = 8200, z = 50000 },
                Heading = 3.14159, Pitch = 0.0 },
        [3] = { Name = "Viper 1-1", CoalitionID = 2, Type = { level1 = 1 },
                Position = { x = 101000, y = 8000, z = 50500 },
                Heading = 0.0, Pitch = 0.0 },
    }
end
LoGetPayloadInfo = function()
    return { Stations = {
        [1] = { count = 2, weapon = { level1 = 4 } },
        [2] = { count = 2, weapon = { level1 = 4 } },
        [3] = { count = 1, weapon = { level1 = 5 } },   -- a bomb: not counted
    } }
end
LoGetLockedTargetInformation = function()
    return { Target = { Distance = 41000.0, Azimuth = 0.05, Elevation = 0.01 } }
end
LoGetTWSInfo = function()
    return { Emitters = {
        [1] = { Azimuth = 0.02, Power = 0.9, Missile = true,
                Type = { Mode = 2 } },
        [2] = { Azimuth = -1.2, Power = 0.3, Missile = false,
                Type = { Mode = 0 } },
    } }
end
local commands = {}
function api.commands() return commands end
LoSetCommand = function(id, value) commands[#commands + 1] = { id, value } end

return api
"""


@unittest.skipIf(lupa is None, "lupa (Lua interpreter) not installed")
class DcsExportTest(unittest.TestCase):
    def setUp(self):
        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.api = self.lua.execute(STUB)
        with open(SCRIPT) as fh:
            chunk = self.lua.eval("function(s) return load(s, 'export') end")(fh.read())
        self.assertIsNotNone(chunk, "the export script does not parse as Lua")
        chunk()
        self.lua.globals().LuaExportStart()

    def _frame(self, t=1.0):
        self.lua.globals().LuaExportActivityNextEvent(t)

    def _last_packet(self):
        packets = self.api.packets()
        self.assertGreater(len(packets), 0, "the export sent no telemetry")
        return packets[len(packets)]

    # ---------------------------------------------------------------- #
    def test_the_packet_parses_with_the_bridge_parser(self):
        self._frame()
        telem = parse_telemetry(self._last_packet().encode())
        self.assertAlmostEqual(telem.t, 12.5)
        self.assertEqual(telem.own.name, "F-16C_50")
        # DCS x=north, y=up, z=east -> ENU (east, north, up).
        self.assertAlmostEqual(telem.own.pos[0], 50000.0)
        self.assertAlmostEqual(telem.own.pos[1], 100000.0)
        self.assertAlmostEqual(telem.own.pos[2], 8000.0)
        self.assertAlmostEqual(telem.own.tas, 250.0)

    def test_it_finds_the_hostile_and_the_friendly_lead(self):
        self._frame()
        telem = parse_telemetry(self._last_packet().encode())
        self.assertIsNotNone(telem.bandit)
        self.assertEqual(telem.bandit.name, "MiG-29S")
        self.assertIsNotNone(telem.lead)
        self.assertEqual(telem.lead.name, "Viper 1-1")

    def test_the_sensor_block_survives_the_round_trip(self):
        self._frame()
        telem = parse_telemetry(self._last_packet().encode())
        sensors = telem.sensors
        self.assertIsNotNone(sensors, "no sensor block in the packet")
        self.assertTrue(sensors.locked)
        self.assertAlmostEqual(sensors.lock_range, 41000.0, delta=0.5)
        self.assertEqual(sensors.missiles, 4)          # the bomb is not counted
        self.assertEqual(len(sensors.threats), 2)
        self.assertTrue(sensors.launch_warning)
        self.assertTrue(sensors.spiked)
        worst = sensors.nearest_threat()
        self.assertTrue(worst.launch)
        self.assertAlmostEqual(worst.az, 1.146, delta=0.01)   # 0.02 rad

    def test_the_packet_is_valid_json_with_no_stray_commas(self):
        self._frame()
        raw = self._last_packet()
        data = json.loads(raw)                 # would raise on a format-string slip
        self.assertEqual(set(data) & {"t", "own", "bandit", "lead", "sensors"},
                         {"t", "own", "bandit", "lead", "sensors"})

    def test_a_module_with_no_radar_still_produces_a_valid_packet(self):
        # Every sensor export missing: the block degrades, the packet does not.
        for name in ("LoGetPayloadInfo", "LoGetLockedTargetInformation",
                     "LoGetTWSInfo"):
            self.lua.globals()[name] = None
        self._frame(2.0)
        telem = parse_telemetry(self._last_packet().encode())
        self.assertIsNotNone(telem.sensors)
        self.assertFalse(telem.sensors.locked)
        self.assertIsNone(telem.sensors.missiles)
        self.assertEqual(telem.sensors.threats, ())

    def test_an_export_that_raises_does_not_take_the_frame_down(self):
        self.lua.execute(
            "LoGetTWSInfo = function() error('module has no radar page') end")
        self._frame(2.0)
        telem = parse_telemetry(self._last_packet().encode())
        self.assertEqual(telem.sensors.threats, ())
        self.assertTrue(telem.sensors.locked)   # the rest of the block survived

    # ---------------------------------------------------------------- #
    def test_it_applies_a_command_line_from_the_bridge(self):
        self.api.push(format_command(
            Controls(pitch=-0.25, roll=0.5, rudder=0.1, thrust=0.8)).decode())
        self._frame()
        ids = [c[1] for c in self.api.commands().values()]
        self.assertIn(2001, ids)   # pitch axis
        self.assertIn(2002, ids)   # roll
        self.assertIn(2004, ids)   # thrust

    def test_weapon_release_is_edge_triggered(self):
        line = format_command(Controls(weapon=1)).decode()
        for frame in range(4):
            self.api.push(line)      # hold the bit down for four frames
            self._frame(float(frame + 1))
        releases = [c for c in self.api.commands().values() if c[1] == 68]
        self.assertEqual(len(releases), 1,
                         "holding the weapon bit emptied the rails")

    def test_releasing_and_re_asserting_fires_again(self):
        fire = format_command(Controls(weapon=1)).decode()
        hold = format_command(Controls(weapon=0)).decode()
        for i, line in enumerate((fire, hold, fire)):
            self.api.push(line)
            self._frame(float(i + 1))
        releases = [c for c in self.api.commands().values() if c[1] == 68]
        self.assertEqual(len(releases), 2)

    def test_an_older_five_field_command_line_still_parses(self):
        self.api.push("-0.2500,0.5000,0.0000,0.8000,1\n")
        self._frame()
        ids = [c[1] for c in self.api.commands().values()]
        self.assertIn(2001, ids)
        self.assertIn(84, ids)     # gun on, from the fifth field


if __name__ == "__main__":
    unittest.main()
