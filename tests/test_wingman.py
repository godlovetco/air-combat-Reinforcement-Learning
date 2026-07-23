import unittest
from types import SimpleNamespace

from dcs_bridge.orders import PilotState, TacticalOrder
from dcs_bridge.wingman import SET_ORDER_TOOL, BrevityParser, WingmanAgent

SITUATION = {
    "own": {"altitude_m": 3000, "heading_deg": 90, "speed_ms": 240},
    "bandit": {"name": "MiG-29", "range_km": 7.5, "bearing_deg": 45,
               "own_aspect_deg": 20, "altitude_m": 3200},
    "mode": "engage",
    "weapons_free": False,
}


class BrevityParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = BrevityParser()

    def order_for(self, text):
        _, orders = self.parser.radio(text, SITUATION)
        return orders[0].order if orders else None

    def test_english_commands(self):
        self.assertEqual(self.order_for("Viper 2, engage the bandit"), "engage")
        self.assertEqual(self.order_for("break right! flare!"), "break_right")
        self.assertEqual(self.order_for("break left"), "break_left")
        self.assertEqual(self.order_for("anchor here"), "anchor")
        self.assertEqual(self.order_for("RTB, mission complete"), "rtb")
        self.assertEqual(self.order_for("weapons free"), "weapons_free")
        self.assertEqual(self.order_for("weapons hold, friendly in the area"), "weapons_hold")

    def test_korean_commands(self):
        self.assertEqual(self.order_for("2번기 교전하라"), "engage")
        self.assertEqual(self.order_for("우측 브레이크!"), "break_right")
        self.assertEqual(self.order_for("기지로 복귀하라"), "rtb")
        self.assertEqual(self.order_for("사격 허가"), "weapons_free")
        self.assertEqual(self.order_for("사격 중지"), "weapons_hold")

    def test_vector_with_heading(self):
        reply, orders = self.parser.radio("vector heading 270", SITUATION)
        self.assertEqual(orders[0].order, "vector")
        self.assertEqual(orders[0].heading_deg, 270.0)
        self.assertIn("270", reply)

    def test_korean_vector(self):
        _, orders = self.parser.radio("헤딩 090 으로 비행", SITUATION)
        self.assertEqual(orders[0].order, "vector")
        self.assertEqual(orders[0].heading_deg, 90.0)

    def test_status_reads_situation(self):
        reply, orders = self.parser.radio("say status", SITUATION)
        self.assertEqual(orders, [])
        self.assertIn("MiG-29", reply)
        self.assertIn("7.5", reply)

    def test_unknown_transmission(self):
        reply, orders = self.parser.radio("what's for dinner", SITUATION)
        self.assertEqual(orders, [])
        self.assertIn("say again", reply.lower())

    def test_cca_formation_commands(self):
        _, orders = self.parser.radio("2, combat spread right side", SITUATION)
        self.assertEqual(orders[0].order, "formation")
        self.assertEqual(orders[0].station, "combat_spread")
        self.assertEqual(orders[0].side, "right")

        _, orders = self.parser.radio("go line abreast on my left", SITUATION)
        self.assertEqual(orders[0].station, "line_abreast")
        self.assertEqual(orders[0].side, "left")

    def test_cca_rejoin_and_scout(self):
        _, orders = self.parser.radio("2, rejoin", SITUATION)
        self.assertEqual(orders[0].order, "rejoin")
        _, orders = self.parser.radio("push ahead and scout", SITUATION)
        self.assertEqual(orders[0].order, "scout")

    def test_cca_leash_beats_weapons_tight(self):
        # "weapons tight" is a leash change, not a weapons_hold order.
        _, orders = self.parser.radio("2, weapons tight", SITUATION)
        self.assertEqual(orders[0].order, "leash")
        self.assertEqual(orders[0].leash, "tight")
        _, orders = self.parser.radio("you're loose", SITUATION)
        self.assertEqual(orders[0].leash, "loose")

    def test_cca_korean_commands(self):
        _, orders = self.parser.radio("전투 전개 대형으로", SITUATION)
        self.assertEqual(orders[0].station, "combat_spread")
        _, orders = self.parser.radio("편대 복귀하라", SITUATION)
        self.assertEqual(orders[0].order, "rejoin")
        _, orders = self.parser.radio("정찰 전진", SITUATION)
        self.assertEqual(orders[0].order, "scout")


class PilotStateTest(unittest.TestCase):
    def test_engage_and_weapons(self):
        state = PilotState()
        self.assertFalse(state.snapshot()["weapons_free"])
        state.apply(TacticalOrder("weapons_free"))
        state.apply(TacticalOrder("engage"))
        snap = state.snapshot()
        self.assertTrue(snap["weapons_free"])
        self.assertEqual(snap["mode"], "engage")

    def test_break_resolves_to_vector(self):
        state = PilotState()
        state.apply(TacticalOrder("break_right"))
        self.assertEqual(state.snapshot()["pending_break"], "right")
        state.resolve_break(current_heading=10.0)
        snap = state.snapshot()
        self.assertEqual(snap["mode"], "vector")
        self.assertAlmostEqual(snap["vector_heading"], 100.0)

    def test_vector_order(self):
        state = PilotState()
        state.apply(TacticalOrder("vector", heading_deg=200.0, altitude_m=5000.0))
        snap = state.snapshot()
        self.assertEqual(snap["mode"], "vector")
        self.assertEqual(snap["vector_heading"], 200.0)
        self.assertEqual(snap["vector_altitude"], 5000.0)

    def test_rtb_then_arrival_anchors(self):
        state = PilotState()
        state.set_home([1000.0, 2000.0, 3000.0])
        state.apply(TacticalOrder("rtb"))
        self.assertEqual(state.snapshot()["mode"], "rtb")
        state.apply_anchor_arrival()
        self.assertEqual(state.snapshot()["mode"], "anchor")
        self.assertEqual(state.home, (1000.0, 2000.0, 3000.0))

    def test_invalid_order_rejected(self):
        with self.assertRaises(ValueError):
            TacticalOrder("self_destruct")

    def test_invalid_side_and_leash_rejected(self):
        with self.assertRaises(ValueError):
            TacticalOrder("formation", side="up")
        with self.assertRaises(ValueError):
            TacticalOrder("leash", leash="strangle")


class CCATeamingTest(unittest.TestCase):
    def test_formation_order_sets_station(self):
        state = PilotState()
        state.apply(TacticalOrder("formation", station="wedge", side="left"))
        snap = state.snapshot()
        self.assertEqual(snap["mode"], "formation")
        self.assertEqual(snap["formation_station"], "wedge")
        self.assertEqual(snap["formation_side"], "left")

    def test_close_leash_cages_weapons_and_pins_formation(self):
        state = PilotState(weapons_free=True, mode="engage")
        state.apply(TacticalOrder("leash", leash="close"))
        snap = state.snapshot()
        self.assertEqual(snap["leash"], "close")
        self.assertFalse(snap["weapons_free"])
        self.assertEqual(snap["mode"], "formation")

    def test_loose_leash_grants_weapons_free(self):
        state = PilotState()
        state.apply(TacticalOrder("leash", leash="loose"))
        self.assertTrue(state.snapshot()["weapons_free"])

    def test_auto_commit_and_rejoin_cycle(self):
        state = PilotState(mode="formation", leash="tight")
        self.assertTrue(state.auto_commit(threat_in_range=True))
        self.assertEqual(state.snapshot()["mode"], "engage")
        # bandit still in range: no rejoin
        self.assertFalse(state.auto_rejoin(threat_gone=False))
        # bandit gone: return to the prior formation mode
        self.assertTrue(state.auto_rejoin(threat_gone=True))
        self.assertEqual(state.snapshot()["mode"], "formation")

    def test_close_leash_blocks_auto_commit(self):
        state = PilotState(mode="formation", leash="close")
        self.assertFalse(state.auto_commit(threat_in_range=True))
        self.assertEqual(state.snapshot()["mode"], "formation")

    def test_lead_commanded_engage_does_not_auto_rejoin(self):
        state = PilotState(mode="engage")  # lead put it here, not auto-commit
        self.assertFalse(state.auto_rejoin(threat_gone=True))
        self.assertEqual(state.snapshot()["mode"], "engage")


class _FakeClient:
    """Stands in for anthropic.Anthropic; scripted responses, no network."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot the messages list: the agent mutates it after the call.
        kwargs["messages"] = list(kwargs["messages"])
        self.requests.append(kwargs)
        return self._responses.pop(0)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool_use(tid, inp):
    return SimpleNamespace(type="tool_use", id=tid, name="set_order", input=inp)


def _response(*content, stop="end_turn"):
    return SimpleNamespace(content=list(content), stop_reason=stop)


class WingmanAgentTest(unittest.TestCase):
    def test_order_flow_with_tool_use(self):
        fake = _FakeClient([
            _response(
                _tool_use("tu_1", {"order": "break_right", "heading_deg": None,
                                   "altitude_m": None, "speed_ms": None}),
                stop="tool_use",
            ),
            _response(_text("2, breaking right!")),
        ])
        agent = WingmanAgent(client=fake)
        reply, orders = agent.radio("break right!", SITUATION)

        self.assertEqual(reply, "2, breaking right!")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order, "break_right")

        # Two API calls: tool round-trip then final text.
        self.assertEqual(len(fake.requests), 2)
        # Tool result was fed back with the matching id.
        tool_result_msg = fake.requests[1]["messages"][-1]
        self.assertEqual(tool_result_msg["content"][0]["tool_use_id"], "tu_1")
        # Strict tool schema goes out on every request.
        self.assertEqual(fake.requests[0]["tools"], [SET_ORDER_TOOL])
        self.assertEqual(fake.requests[0]["model"], "claude-opus-4-8")

    def test_status_needs_no_tool(self):
        fake = _FakeClient([
            _response(_text("2 has tally, MiG-29, 7.5 kilometers, angels 10.")),
        ])
        agent = WingmanAgent(client=fake)
        reply, orders = agent.radio("2, say status", SITUATION)
        self.assertEqual(orders, [])
        self.assertIn("tally", reply)
        self.assertEqual(len(fake.requests), 1)

    def test_bad_tool_input_reported_not_crashing(self):
        fake = _FakeClient([
            _response(
                _tool_use("tu_9", {"order": "warp_speed", "heading_deg": None,
                                   "altitude_m": None, "speed_ms": None}),
                stop="tool_use",
            ),
            _response(_text("2, unable.")),
        ])
        agent = WingmanAgent(client=fake)
        reply, orders = agent.radio("engage warp drive", SITUATION)
        self.assertEqual(orders, [])
        self.assertEqual(reply, "2, unable.")
        result = fake.requests[1]["messages"][-1]["content"][0]
        self.assertIn("rejected", result["content"])

    def test_history_is_trimmed(self):
        responses = [_response(_text(f"copy {i}")) for i in range(40)]
        fake = _FakeClient(responses)
        agent = WingmanAgent(client=fake)
        for i in range(40):
            agent.radio(f"radio check {i}", SITUATION)
        user_texts = [m for m in agent.messages
                      if m["role"] == "user" and isinstance(m["content"], str)]
        self.assertLessEqual(len(user_texts), 12)


if __name__ == "__main__":
    unittest.main()
