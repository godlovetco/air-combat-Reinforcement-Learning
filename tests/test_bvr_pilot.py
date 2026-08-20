import collections
import unittest

from dcs_bridge import geometry as geo
from dcs_bridge.bvr_env import BVRSimEnv
from dcs_bridge.bvr_pilot import (CRANK_DEG, NOTCH_DEG, TimelinePilot,
                                  _bearing, _nearer_beam, situation_from_env)


def _sit(heading=0.0, target_bearing=0.0, threat=None, shot=False, owed=0.0):
    return {
        "own": {"heading_deg": heading, "gamma_deg": 0.0, "tas": 250.0,
                "altitude_m": 8000.0},
        "target_bearing_deg": target_bearing,
        "bvr": {"threat": threat, "shot_in_flight": shot,
                "support_owed_s": owed, "missiles": 2, "locked": True,
                "spiked": False, "wez": {}},
    }


class HelpersTest(unittest.TestCase):
    def test_bearing(self):
        self.assertAlmostEqual(_bearing([0.0, 0.0, 0.0], [0.0, 10.0, 0.0]), 0.0)
        self.assertAlmostEqual(_bearing([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]), 90.0)

    def test_nearer_beam_picks_the_shorter_turn(self):
        # Threat due north; we are heading east, so the beam to take is 090.
        self.assertAlmostEqual(_nearer_beam(0.0, 90.0), 90.0)
        # Heading west: the other beam.
        self.assertAlmostEqual(_nearer_beam(0.0, 270.0), 270.0)

    def test_both_beams_are_ninety_off_the_threat(self):
        for psi in (0.0, 45.0, 135.0, 250.0):
            beam = _nearer_beam(30.0, psi)
            self.assertAlmostEqual(abs(geo.heading_error(beam, 30.0)), NOTCH_DEG,
                                   places=6)


class DesiredHeadingTest(unittest.TestCase):
    def setUp(self):
        self.pilot = TimelinePilot()

    def test_commit_when_nothing_is_in_the_air(self):
        self.assertAlmostEqual(
            self.pilot.desired_heading(_sit(heading=10.0, target_bearing=45.0)),
            45.0)

    def test_support_cranks_off_the_target(self):
        want = self.pilot.desired_heading(
            _sit(heading=0.0, target_bearing=0.0, shot=True, owed=20.0))
        self.assertAlmostEqual(abs(geo.heading_error(want, 0.0)), CRANK_DEG,
                               places=6)

    def test_the_crank_goes_the_shorter_way(self):
        # Already turned left of the target: crank left, not back through it.
        want = self.pilot.desired_heading(
            _sit(heading=-40.0, target_bearing=0.0, shot=True, owed=20.0))
        self.assertLess(geo.heading_error(want, 0.0), 0.0)
        want = self.pilot.desired_heading(
            _sit(heading=40.0, target_bearing=0.0, shot=True, owed=20.0))
        self.assertGreater(geo.heading_error(want, 0.0), 0.0)

    def test_no_crank_once_the_shot_is_pitbull(self):
        want = self.pilot.desired_heading(
            _sit(heading=0.0, target_bearing=30.0, shot=True, owed=0.0))
        self.assertAlmostEqual(want, 30.0)

    def test_defend_beams_the_missile(self):
        # Threat 90 degrees right of the nose while we head north: beam is 000+90
        # relative to the threat bearing 090 -> either 000 or 180; the shorter
        # turn from north is 000.
        threat = {"off_nose_deg": 90.0, "active": True, "seconds": 12.0,
                  "notch_depth": 1.0}
        want = self.pilot.desired_heading(
            _sit(heading=0.0, target_bearing=0.0, threat=threat))
        self.assertAlmostEqual(abs(geo.heading_error(want, 90.0)), NOTCH_DEG,
                               places=6)

    def test_defend_outranks_support(self):
        threat = {"off_nose_deg": 90.0, "active": True, "seconds": 12.0,
                  "notch_depth": 1.0}
        want = self.pilot.desired_heading(
            _sit(heading=0.0, target_bearing=0.0, threat=threat, shot=True,
                 owed=30.0))
        # A crank would be 50 degrees off the target; a beam is 90 off the
        # threat.  It took the beam.
        self.assertAlmostEqual(abs(geo.heading_error(want, 90.0)), NOTCH_DEG,
                               places=6)

    def test_an_inactive_threat_does_not_trigger_defend(self):
        threat = {"off_nose_deg": 90.0, "active": False, "seconds": 40.0,
                  "notch_depth": 1.0}
        self.assertAlmostEqual(
            self.pilot.desired_heading(
                _sit(heading=0.0, target_bearing=20.0, threat=threat)),
            20.0)


class ActTest(unittest.TestCase):
    def test_returns_a_valid_index_for_both_action_sets(self):
        pilot = TimelinePilot()
        sit = _sit(heading=0.0, target_bearing=90.0)
        self.assertIn(pilot.act(sit, "legacy"), range(9))
        self.assertIn(pilot.act(sit, "energy"), range(27))

    def test_it_turns_toward_the_desired_heading(self):
        pilot = TimelinePilot()
        sit = _sit(heading=0.0, target_bearing=90.0)
        cands = geo.candidate_actions(250.0, 0.0, 0.0)
        chosen = cands[pilot.act(sit, "legacy")]
        self.assertGreater(geo.heading_error(chosen[2], 0.0), 0.0)  # turned right

    def test_it_prefers_the_flattest_climb_among_equal_headings(self):
        pilot = TimelinePilot()
        sit = _sit(heading=0.0, target_bearing=90.0)
        chosen = geo.candidate_actions(250.0, 0.0, 0.0)[pilot.act(sit, "legacy")]
        self.assertAlmostEqual(chosen[1], 0.0)


class EngagementTest(unittest.TestCase):
    """The reason this module exists: it has to actually win."""

    def _run(self, pick, opponent, n=60, seed=4242):
        env = BVRSimEnv(seed=seed, opponent=opponent, shaping=0.0)
        c = collections.Counter()
        for _ in range(n):
            env.reset()
            done, info = False, {}
            while not done:
                _o, _r, done, info = env.step(pick(env))
            c[info["outcome"]] += 1
        return c

    def _hot(self, env):
        want = _bearing(env.pos_r, env.pos_b)
        cands = geo.candidate_actions(*env.act_r, action_set=env.action_set,
                                      dt=env.dt)
        return min(range(len(cands)),
                   key=lambda i: (abs(geo.heading_error(want, cands[i][2])),
                                  abs(cands[i][1])))

    def _timeline(self, env):
        return TimelinePilot().act(situation_from_env(env), env.action_set, env.dt)

    def test_it_converts_a_straight_flier(self):
        c = self._run(self._timeline, "straight")
        self.assertGreater(c["win"] / 60, 0.85)

    def test_it_converts_a_pursuing_bandit(self):
        c = self._run(self._timeline, "pursuit")
        self.assertGreater(c["win"] / 60, 0.85)

    def test_it_survives_an_evasive_bandit_that_kills_the_naive_agent(self):
        naive = self._run(self._hot, "evasive")
        timeline = self._run(self._timeline, "evasive")
        self.assertGreater(naive["loss"], 30)          # pressing in gets you shot
        self.assertLess(timeline["loss"], 5)           # the timeline does not
        self.assertGreater(timeline["win"], naive["win"])

    def test_it_does_not_lose_to_the_ace_bandit(self):
        c = self._run(self._timeline, "ace")
        self.assertEqual(c["loss"], 0)

    def test_situation_from_env_works_for_either_side(self):
        env = BVRSimEnv(seed=7, opponent="ace", shaping=0.0)
        env.reset()
        for who in ("agent", "bandit"):
            sit = situation_from_env(env, who)
            self.assertIn("target_bearing_deg", sit)
            self.assertIn("threat", sit["bvr"])
            self.assertIn(TimelinePilot().act(sit, env.action_set, env.dt),
                          range(9))


if __name__ == "__main__":
    unittest.main()
