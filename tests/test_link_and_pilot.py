import json
import math
import socket
import threading
import unittest

from dcs_bridge.autopilot import Autopilot, AutopilotConfig, Controls
from dcs_bridge.link import DCSLink, format_command, parse_telemetry
from dcs_bridge.predictor import TurnRatePredictor


def make_packet(t, own_pos=(1000.0, 2000.0, 3000.0), bandit=True):
    data = {
        "t": t,
        "own": {
            "name": "UCAV", "px": own_pos[0], "py": own_pos[1], "pz": own_pos[2],
            "heading": 45.0, "pitch": 2.0, "bank": -5.0, "tas": 240.0, "vv": 3.0,
        },
    }
    if bandit:
        data["bandit"] = {
            "name": "MiG-29", "px": 5000.0, "py": 8000.0, "pz": 3500.0,
            "heading": 180.0, "pitch": 0.0,
        }
    return json.dumps(data).encode()


class ProtocolTest(unittest.TestCase):
    def test_parse_telemetry(self):
        telem = parse_telemetry(make_packet(12.5))
        self.assertEqual(telem.t, 12.5)
        self.assertEqual(telem.own.name, "UCAV")
        self.assertEqual(telem.own.pos, (1000.0, 2000.0, 3000.0))
        self.assertEqual(telem.bandit.name, "MiG-29")

    def test_parse_without_bandit(self):
        telem = parse_telemetry(make_packet(1.0, bandit=False))
        self.assertIsNone(telem.bandit)
        self.assertIsNone(telem.lead)

    def test_parse_lead_datalink(self):
        data = json.loads(make_packet(2.0, bandit=False).decode())
        data["lead"] = {
            "name": "Viper 1", "px": 100.0, "py": 200.0, "pz": 3000.0,
            "heading": 30.0, "pitch": 1.0, "tas": 255.0,
        }
        telem = parse_telemetry(json.dumps(data).encode())
        self.assertIsNotNone(telem.lead)
        self.assertEqual(telem.lead.name, "Viper 1")
        self.assertEqual(telem.lead.pos, (100.0, 200.0, 3000.0))
        self.assertEqual(telem.lead.tas, 255.0)

    def test_lead_tas_optional(self):
        data = json.loads(make_packet(2.0, bandit=False).decode())
        data["lead"] = {"name": "Eagle", "px": 0.0, "py": 0.0, "pz": 3000.0}
        telem = parse_telemetry(json.dumps(data).encode())
        self.assertIsNone(telem.lead.tas)

    def test_format_command_matches_lua_pattern(self):
        line = format_command(
            Controls(pitch=-0.25, roll=0.5, rudder=0.0, thrust=0.8, trigger=1)
        ).decode()
        # Same pattern the Lua side uses to parse commands.
        self.assertRegex(
            line, r"^-?[\d.]+,-?[\d.]+,-?[\d.]+,-?[\d.]+,[01]\n$"
        )
        parts = line.strip().split(",")
        self.assertAlmostEqual(float(parts[0]), -0.25)
        self.assertEqual(parts[4], "1")

    def test_udp_roundtrip_with_fake_dcs(self):
        """Python link <-> a fake DCS export script over real UDP sockets."""
        # Fake DCS: sends telemetry to the link, listens for commands.
        cmd_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd_rx.bind(("127.0.0.1", 0))
        cmd_rx.settimeout(2.0)
        cmd_port = cmd_rx.getsockname()[1]

        link = DCSLink(
            listen_host="127.0.0.1", telemetry_port=0,
            dcs_host="127.0.0.1", command_port=cmd_port, timeout=2.0,
        )
        telem_port = link.rx.getsockname()[1]

        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.sendto(make_packet(3.25), ("127.0.0.1", telem_port))

        telem = link.receive()
        self.assertIsNotNone(telem)
        self.assertEqual(telem.t, 3.25)

        link.send(Controls(pitch=0.1, roll=-0.2, rudder=0.0, thrust=0.7))
        payload, _ = cmd_rx.recvfrom(1024)
        self.assertTrue(payload.decode().startswith("0.1000,-0.2000,"))

        tx.close()
        cmd_rx.close()
        link.close()


class AutopilotTest(unittest.TestCase):
    def test_turns_toward_commanded_heading(self):
        ap = Autopilot(AutopilotConfig(invert_pitch=False, invert_roll=False))
        c = ap.command(t=1.0, pitch_deg=0, bank_deg=0, heading_deg=0,
                       tas=240, gamma_cmd_deg=0, psi_cmd_deg=90, v_cmd=250)
        self.assertGreater(c.roll, 0.0, "should roll right toward 090")
        c = ap.command(t=1.05, pitch_deg=0, bank_deg=0, heading_deg=0,
                       tas=240, gamma_cmd_deg=0, psi_cmd_deg=270, v_cmd=250)
        self.assertLess(c.roll, 0.0, "should roll left toward 270")

    def test_pulls_up_for_climb_command(self):
        ap = Autopilot(AutopilotConfig(invert_pitch=False))
        c = ap.command(t=1.0, pitch_deg=0, bank_deg=0, heading_deg=0,
                       tas=240, gamma_cmd_deg=30, psi_cmd_deg=0, v_cmd=250)
        self.assertGreater(c.pitch, 0.0)

    def test_throttle_responds_to_speed_error(self):
        ap = Autopilot(AutopilotConfig())
        slow = ap.command(t=1.0, pitch_deg=0, bank_deg=0, heading_deg=0,
                          tas=150, gamma_cmd_deg=0, psi_cmd_deg=0, v_cmd=250)
        ap2 = Autopilot(AutopilotConfig())
        fast = ap2.command(t=1.0, pitch_deg=0, bank_deg=0, heading_deg=0,
                           tas=350, gamma_cmd_deg=0, psi_cmd_deg=0, v_cmd=250)
        self.assertGreater(slow.thrust, fast.thrust)
        self.assertTrue(0.0 <= slow.thrust <= 1.0)
        self.assertTrue(0.0 <= fast.thrust <= 1.0)

    def test_axes_bounded(self):
        ap = Autopilot(AutopilotConfig())
        c = ap.command(t=1.0, pitch_deg=-80, bank_deg=170, heading_deg=0,
                       tas=100, gamma_cmd_deg=70, psi_cmd_deg=180, v_cmd=400)
        for v in (c.pitch, c.roll, c.rudder):
            self.assertTrue(-1.0 <= v <= 1.0)


class PredictorTest(unittest.TestCase):
    def test_straight_line_prediction(self):
        pred = TurnRatePredictor()
        for i in range(20):
            t = i * 0.1
            pred.update(t, (100.0 * t, 0.0, 3000.0))  # 100 m/s due east
        p = pred.predict(2.0)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[0], 100.0 * 1.9 + 200.0, delta=15.0)
        self.assertAlmostEqual(p[1], 0.0, delta=5.0)

    def test_turning_prediction_beats_linear(self):
        """On a constant-rate turn the arc model must beat extrapolating straight."""
        pred = TurnRatePredictor()
        speed, rate = 200.0, math.radians(9.0)  # 9 deg/s turn

        def pos(t):
            r = speed / rate
            return (r * math.sin(rate * t), r * (1.0 - math.cos(rate * t)), 3000.0)

        for i in range(26):
            pred.update(i * 0.1, pos(i * 0.1))
        t_now, horizon = 2.5, 3.0
        truth = pos(t_now + horizon)
        predicted = pred.predict(horizon)
        self.assertIsNotNone(predicted)

        # Linear extrapolation from the last two samples.
        p1, p0 = pos(t_now), pos(t_now - 0.1)
        vx, vy = (p1[0] - p0[0]) / 0.1, (p1[1] - p0[1]) / 0.1
        linear = (p1[0] + vx * horizon, p1[1] + vy * horizon, 3000.0)

        err_pred = math.dist(predicted, truth)
        err_linear = math.dist(linear, truth)
        self.assertLess(err_pred, err_linear * 0.5,
                        f"arc {err_pred:.0f} m vs linear {err_linear:.0f} m")

    def test_intercept_point_leads_target(self):
        pred = TurnRatePredictor()
        for i in range(20):
            t = i * 0.1
            pred.update(t, (5000.0 - 200.0 * t, 5000.0, 3000.0))  # heading west
        lead = pred.intercept_point(own_pos=(0.0, 0.0, 3000.0), own_speed=250.0)
        self.assertIsNotNone(lead)
        self.assertLess(lead[0], 5000.0 - 200.0 * 1.9, "lead point must be ahead of the bandit")


if __name__ == "__main__":
    unittest.main()
