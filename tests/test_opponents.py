import math
import unittest

from dcs_bridge import geometry as geo
from dcs_bridge.sim_env import UCAVSimEnv


def _bearing(src, dst):
    return geo.wrap_heading(math.degrees(math.atan2(dst[0] - src[0], dst[1] - src[1])))


class OpponentTest(unittest.TestCase):
    def test_straight_is_unchanged_default(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=1)  # default opponent
        heading0 = env.act_b[2]
        for _ in range(5):
            env.step(4)  # agent holds; bandit must not turn
        self.assertEqual(env.act_b[2], heading0)

    def test_unknown_opponent_rejected(self):
        with self.assertRaises(ValueError):
            UCAVSimEnv(opponent="kamikaze")

    def test_pursuit_turns_toward_the_agent(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=2, opponent="pursuit")
        # Place the agent off the bandit's nose so a turn is required.
        env.pos_b = [100_000.0, 100_000.0, 3_000.0]
        env.pos_r = [130_000.0, 100_000.0, 3_000.0]  # due east of the bandit
        env.act_b = [250.0, 0.0, 0.0]                # bandit pointing north
        want = _bearing(env.pos_b, env.pos_r)        # ~090
        before = abs(geo.heading_error(want, env.act_b[2]))
        env.step(4)
        after = abs(geo.heading_error(want, env.act_b[2]))
        self.assertLess(after, before)               # closed the pointing error
        self.assertLessEqual(before - after, 10.0 + 1e-6)  # capped turn rate

    def test_evasive_breaks_when_threatened_from_behind(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=3, opponent="evasive")
        # Agent 2 km directly behind the north-bound bandit, on its tail.
        env.pos_b = [100_000.0, 100_000.0, 3_000.0]
        env.act_b = [250.0, 0.0, 0.0]                # heading north
        env.pos_r = [100_000.0, 98_000.0, 3_000.0]   # 2 km south (astern)
        env.act_r = [250.0, 0.0, 0.0]                # also north (pursuing)
        env.step(4)
        # A threatened bandit breaks toward the beam (~a 10 deg bite this step).
        self.assertGreater(abs(geo.heading_error(env.act_b[2], 0.0)), 0.0)

    def test_evasive_flies_straight_when_not_threatened(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=4, opponent="evasive")
        env.pos_b = [100_000.0, 100_000.0, 3_000.0]
        env.act_b = [250.0, 0.0, 0.0]
        env.pos_r = [180_000.0, 100_000.0, 3_000.0]  # 80 km away, not a threat
        heading0 = env.act_b[2]
        env.step(4)
        self.assertEqual(env.act_b[2], heading0)

    def test_mixed_resolves_per_episode(self):
        env = UCAVSimEnv(randomize=True, shaping=0.0, seed=5, opponent="mixed")
        seen = set()
        for _ in range(30):
            env.reset()
            seen.add(env._episode_opponent)
        self.assertTrue(seen <= {"straight", "pursuit", "evasive"})
        self.assertGreater(len(seen), 1)  # actually varies

    def test_reactive_episode_runs_to_completion(self):
        env = UCAVSimEnv(randomize=True, shaping=0.05, seed=6, opponent="mixed")
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < env.max_steps + 1:
            obs, _r, done, info = env.step(4)
            steps += 1
        self.assertTrue(done)
        self.assertIn(info["outcome"], ("win", "loss", "out_of_bounds", "timeout"))


if __name__ == "__main__":
    unittest.main()
