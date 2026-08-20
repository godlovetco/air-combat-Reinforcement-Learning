import unittest
from types import SimpleNamespace

from dcs_bridge.wso import WSOAdvisor, WSOAgent, make_wso


def bandit(range_m, own_aspect=180.0, bandit_aspect=180.0, bearing=90.0, alt=3000.0,
           name="MiG-29"):
    return {
        "name": name, "range_m": range_m, "own_aspect_deg": own_aspect,
        "bandit_aspect_deg": bandit_aspect, "bearing_deg": bearing, "altitude_m": alt,
    }


def sit(t=0.0, alt=6000.0, tas=250.0, vv=0.0, bank=0.0, gamma=0.0, heading=0.0,
        b=None, lead=None, weapons_free=False, recommend=None):
    return {
        "t": t,
        "own": {"altitude_m": alt, "tas": tas, "vv": vv, "bank_deg": bank,
                "gamma_deg": gamma, "heading_deg": heading},
        "bandit": b, "lead": lead, "weapons_free": weapons_free, "recommend": recommend,
    }


class WSOAdvisorTest(unittest.TestCase):
    def test_contact_and_no_joy_are_edge_triggered(self):
        wso = WSOAdvisor(lang="en")
        first = wso.advise(sit(t=0.0, b=bandit(9000.0)))
        self.assertIsNotNone(first)
        self.assertIn("Contact", first)
        # bandit gone -> "no joy" fires immediately
        gone = wso.advise(sit(t=1.0, b=None))
        self.assertIsNotNone(gone)
        self.assertIn("No joy", gone)

    def test_guns_call_in_parameters(self):
        wso = WSOAdvisor(lang="en")
        wso.advise(sit(t=0.0, b=bandit(9000.0)))  # consume the contact call
        line = wso.advise(sit(t=0.5, b=bandit(800.0, own_aspect=2.0),
                              weapons_free=True))
        self.assertIn("guns", line.lower())

    def test_break_when_bandit_nose_on_and_close(self):
        wso = WSOAdvisor(lang="en")
        wso.advise(sit(t=0.0, b=bandit(4000.0, bandit_aspect=10.0)))  # contact
        # closing fast, inside merge range, bandit nose-on -> break
        line = wso.advise(sit(t=1.0, b=bandit(2000.0, bandit_aspect=10.0)))
        self.assertIn("Break", line)

    def test_altitude_safety_is_top_priority(self):
        wso = WSOAdvisor(lang="en")
        line = wso.advise(sit(t=0.0, alt=150.0, vv=-20.0, b=bandit(3000.0)))
        self.assertIn("Altitude", line)

    def test_maneuver_recommendation_is_rendered(self):
        wso = WSOAdvisor(lang="en")
        wso.advise(sit(t=0.0, b=bandit(6000.0)))  # contact
        line = wso.advise(sit(t=0.5, heading=0.0, b=bandit(6000.0, own_aspect=40.0),
                              recommend={"heading_deg": 90.0, "gamma_deg": 10.0}))
        # right turn to 090, nose up
        self.assertIn("right", line.lower())
        self.assertIn("090", line)
        self.assertIn("nose up", line.lower())

    def test_bra_respects_cooldown(self):
        wso = WSOAdvisor(lang="en")
        wso.advise(sit(t=0.0, b=bandit(9000.0)))  # contact (a BRA-key call)
        # immediately after, the BRA key is on cooldown -> nothing new
        self.assertIsNone(wso.advise(sit(t=0.2, b=bandit(9000.0))))
        # after the cooldown a fresh BRA is allowed
        line = wso.advise(sit(t=11.0, b=bandit(9000.0)))
        self.assertIsNotNone(line)
        self.assertIn("BRA", line)

    def test_korean_output(self):
        wso = WSOAdvisor(lang="ko")
        line = wso.advise(sit(t=0.0, b=bandit(9000.0)))
        self.assertIn("컨택", line)

    def test_station_call_when_off_formation(self):
        wso = WSOAdvisor(lang="en")
        line = wso.advise(sit(t=0.0, lead={"name": "1", "range_m": 3000.0,
                                           "station_error_m": 900.0}))
        self.assertIn("station", line.lower())


class MakeWSOTest(unittest.TestCase):
    def test_rule_based_default(self):
        self.assertIsInstance(make_wso(use_llm=False), WSOAdvisor)

    def test_llm_with_fake_client(self):
        fake = SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Come right to 090.")])))
        agent = WSOAgent(client=fake, lang="en")
        self.assertEqual(agent.advise(sit(b=bandit(5000.0))), "Come right to 090.")

    def test_llm_empty_reply_is_none(self):
        fake = SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                content=[SimpleNamespace(type="text", text="   ")])))
        agent = WSOAgent(client=fake, lang="en")
        self.assertIsNone(agent.advise(sit(b=None)))


if __name__ == "__main__":
    unittest.main()
