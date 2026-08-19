import unittest

from dcs_bridge import bvr
from dcs_bridge import geometry as geo
from dcs_bridge.bvr_env import BVRSimEnv, wso_situation
from dcs_bridge.wso import WSOAdvisor, _clock


class ClockTest(unittest.TestCase):
    def test_off_nose_to_clock(self):
        self.assertEqual(_clock(0.0), 12)
        self.assertEqual(_clock(90.0), 3)
        self.assertEqual(_clock(-90.0), 9)
        self.assertEqual(_clock(180.0), 6)
        self.assertEqual(_clock(-180.0), 6)
        self.assertEqual(_clock(30.0), 1)
        self.assertEqual(_clock(-30.0), 11)


class WsoSituationTest(unittest.TestCase):
    def setUp(self):
        self.env = BVRSimEnv(seed=50, randomize=False, shaping=0.0,
                             opponent="straight")
        self.env.reset()

    def test_block_shape_at_the_merge_start(self):
        sit = wso_situation(self.env)
        self.assertTrue(sit["locked"])
        self.assertIsNone(sit["threat"])
        self.assertFalse(sit["shot_in_flight"])
        self.assertEqual(sit["missiles"], self.env.loadout)
        self.assertIn("in_nez", sit["wez"])
        self.assertEqual(sit["support_owed_s"], 0.0)

    def test_it_reports_an_incoming_missile(self):
        m = bvr.Missile.launch(self.env.spec, "bandit",
                               [self.env.pos_r[0] + 20_000.0, self.env.pos_r[1],
                                self.env.pos_r[2]], [900.0, 0.0, 270.0])
        m.active = True
        self.env.in_flight.append(m)
        self.env.act_r = [250.0, 0.0, 0.0]         # we point north
        threat = wso_situation(self.env)["threat"]
        self.assertIsNotNone(threat)
        self.assertTrue(threat["active"])
        self.assertAlmostEqual(threat["off_nose_deg"], 90.0, places=3)  # 3 o'clock
        self.assertAlmostEqual(threat["range_m"], 20_000.0, delta=1.0)
        self.assertAlmostEqual(threat["notch_depth"], 0.0, places=6)    # beaming it

    def test_it_reports_support_owed_on_our_own_shot(self):
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 40_000.0, 8000.0]
        self.env.act_b = [250.0, 0.0, 180.0]
        self.env._update_tracks()
        self.assertTrue(self.env._try_launch("agent"))
        sit = wso_situation(self.env)
        self.assertTrue(sit["shot_in_flight"])
        self.assertGreater(sit["support_owed_s"], 0.0)

    def test_the_bandit_side_can_be_asked_too(self):
        sit = wso_situation(self.env, "bandit")
        self.assertIn("locked", sit)
        self.assertEqual(sit["missiles"], self.env.missiles_b)


def _sit(t=0.0, **bvr_block):
    base = {
        "t": t,
        "own": {"altitude_m": 8000.0, "tas": 250.0, "vv": 0.0,
                "bank_deg": 0.0, "heading_deg": 0.0},
        "bandit": None,
        "bvr": bvr_block,
    }
    return base


def _wez(rng, in_env=False, in_nez=False, threatened=False):
    return {"range": rng, "rmax": 60_000.0, "rtr": 20_000.0, "rne": 20_000.0,
            "rmin": 1_500.0, "time_of_flight": 30.0, "in_envelope": in_env,
            "in_nez": in_nez, "threatened": threatened}


class WsoBvrCallsTest(unittest.TestCase):
    def setUp(self):
        self.wso = WSOAdvisor(lang="en")

    def test_no_bvr_block_is_a_no_op(self):
        sit = {"t": 0.0, "own": {"altitude_m": 8000.0, "tas": 250.0, "vv": 0.0,
                                 "bank_deg": 0.0, "heading_deg": 0.0},
               "bandit": None}
        self.assertIsNone(self.wso.advise(sit))

    def test_defend_names_the_clock_and_the_turn(self):
        sit = _sit(threat={"off_nose_deg": 90.0, "seconds": 12.0, "active": True,
                           "notch_depth": 1.0},
                   wez=_wez(30_000.0), missiles=2, shot_in_flight=False,
                   support_owed_s=0.0, locked=True, spiked=True)
        line = self.wso.advise(sit)
        self.assertIn("Defend", line)
        self.assertIn("3 o'clock", line)
        self.assertIn("notch left", line)   # threat on the right -> turn left
        self.assertIn("12 seconds", line)

    def test_already_in_the_notch_says_hold_it(self):
        sit = _sit(threat={"off_nose_deg": 90.0, "seconds": 10.0, "active": True,
                           "notch_depth": 0.0},
                   wez=_wez(30_000.0), missiles=2, shot_in_flight=False,
                   support_owed_s=0.0, locked=True, spiked=True)
        line = self.wso.advise(sit)
        self.assertIn("notch", line.lower())
        self.assertIn("hold", line.lower())

    def test_a_distant_missile_is_a_setup_call_not_a_defend(self):
        sit = _sit(threat={"off_nose_deg": -60.0, "seconds": 60.0, "active": True,
                           "notch_depth": 1.0},
                   wez=_wez(60_000.0), missiles=2, shot_in_flight=False,
                   support_owed_s=0.0, locked=True, spiked=True)
        line = self.wso.advise(sit)
        self.assertNotIn("Defend", line)
        self.assertIn("set up the notch", line)

    def test_defending_outranks_shooting(self):
        sit = _sit(threat={"off_nose_deg": 45.0, "seconds": 8.0, "active": True,
                           "notch_depth": 0.9},
                   wez=_wez(15_000.0, in_env=True, in_nez=True), missiles=2,
                   shot_in_flight=False, support_owed_s=0.0, locked=True,
                   spiked=True)
        self.assertIn("Defend", self.wso.advise(sit))

    def test_flying_into_the_ground_still_outranks_defending(self):
        sit = _sit(threat={"off_nose_deg": 45.0, "seconds": 8.0, "active": True,
                           "notch_depth": 0.9},
                   wez=_wez(15_000.0), missiles=2, shot_in_flight=False,
                   support_owed_s=0.0, locked=True, spiked=True)
        sit["own"]["altitude_m"] = 200.0
        sit["own"]["vv"] = -50.0
        self.assertIn("Altitude", self.wso.advise(sit))

    def test_support_call_carries_the_seconds_owed(self):
        sit = _sit(threat=None, wez=_wez(40_000.0), missiles=1,
                   shot_in_flight=True, support_owed_s=14.0, locked=True,
                   spiked=False)
        line = self.wso.advise(sit)
        self.assertIn("Support 14 seconds", line)
        self.assertIn("crank", line)

    def test_pitbull_releases_the_pilot(self):
        sit = _sit(threat=None, wez=_wez(20_000.0), missiles=1,
                   shot_in_flight=True, support_owed_s=0.0, locked=True,
                   spiked=False)
        self.assertIn("Pitbull", self.wso.advise(sit))

    def test_shoot_only_inside_the_no_escape_zone(self):
        nez = _sit(threat=None, wez=_wez(15_000.0, in_env=True, in_nez=True),
                   missiles=2, shot_in_flight=False, support_owed_s=0.0,
                   locked=True, spiked=False)
        self.assertIn("shoot", self.wso.advise(nez).lower())

        far = WSOAdvisor(lang="en").advise(
            _sit(threat=None, wez=_wez(45_000.0, in_env=True, in_nez=False),
                 missiles=2, shot_in_flight=False, support_owed_s=0.0,
                 locked=True, spiked=False))
        self.assertIn("he can run", far)

    def test_winchester(self):
        sit = _sit(threat=None, wez=_wez(30_000.0), missiles=0,
                   shot_in_flight=False, support_owed_s=0.0, locked=True,
                   spiked=False)
        self.assertIn("Winchester", self.wso.advise(sit))

    def test_korean_calls(self):
        wso = WSOAdvisor(lang="ko")
        sit = _sit(threat={"off_nose_deg": -90.0, "seconds": 10.0, "active": True,
                           "notch_depth": 1.0},
                   wez=_wez(20_000.0), missiles=2, shot_in_flight=False,
                   support_owed_s=0.0, locked=True, spiked=True)
        line = wso.advise(sit)
        self.assertIn("디펜드", line)
        self.assertIn("9시", line)
        self.assertIn("우로", line)   # threat on the left -> notch right

    def test_calls_are_rate_limited(self):
        sit = _sit(threat=None, wez=_wez(15_000.0, in_env=True, in_nez=True),
                   missiles=2, shot_in_flight=False, support_owed_s=0.0,
                   locked=True, spiked=False)
        self.assertIsNotNone(self.wso.advise(sit))
        self.assertIsNone(self.wso.advise(sit))          # on cooldown
        sit["t"] = 30.0
        self.assertIsNotNone(self.wso.advise(sit))

    def test_it_runs_against_a_live_engagement(self):
        env = BVRSimEnv(seed=51, opponent="ace", shaping=0.0)
        env.reset()
        wso = WSOAdvisor(lang="en")
        lines, done, t = [], False, 0.0
        while not done:
            _o, _r, done, _i = env.step(4)
            t += 1.0
            sit = _sit(t=t, **wso_situation(env))
            line = wso.advise(sit)
            if line:
                lines.append(line)
        self.assertTrue(lines)
        self.assertTrue(all(isinstance(x, str) and x for x in lines))


if __name__ == "__main__":
    unittest.main()


class ChatterTest(unittest.TestCase):
    def test_winchester_waits_for_the_last_shot_to_finish(self):
        wso = WSOAdvisor(lang="en")
        still_guiding = _sit(threat=None, wez=_wez(30_000.0), missiles=0,
                             shot_in_flight=True, support_owed_s=5.0,
                             locked=True, spiked=False)
        line = wso.advise(still_guiding)
        self.assertNotIn("Winchester", line or "")
        spent = _sit(t=100.0, threat=None, wez=_wez(30_000.0), missiles=0,
                     shot_in_flight=False, support_owed_s=0.0, locked=True,
                     spiked=False)
        self.assertIn("Winchester", wso.advise(spent))

    def test_a_long_engagement_does_not_repeat_one_call_endlessly(self):
        wso = WSOAdvisor(lang="en")
        said = []
        for step in range(200):
            sit = _sit(t=float(step), threat=None, wez=_wez(30_000.0),
                       missiles=0, shot_in_flight=False, support_owed_s=0.0,
                       locked=True, spiked=False)
            line = wso.advise(sit)
            if line:
                said.append(line)
        # 200 seconds of nothing happening should not be 200 calls.
        self.assertLess(len(said), 8)
