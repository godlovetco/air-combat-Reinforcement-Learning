import unittest

import numpy as np

from dcs_bridge.geometry import INPUT_DIM
from dcs_bridge.sim_env import UCAVSimEnv


class SimEnvTest(unittest.TestCase):
    def test_episode_terminates(self):
        env = UCAVSimEnv(max_steps=50, randomize=False, seed=0)
        obs = env.reset()
        self.assertEqual(obs.shape, (INPUT_DIM,))
        done = False
        steps = 0
        while not done:
            obs, reward, done, info = env.step(4)  # hold straight & level
            steps += 1
            self.assertTrue(np.isfinite(obs).all())
        self.assertLessEqual(steps, 50)
        self.assertIn(info["outcome"], {"win", "loss", "timeout", "out_of_bounds"})

    def test_step_after_done_raises(self):
        env = UCAVSimEnv(max_steps=1, randomize=False)
        env.reset()
        env.step(4)
        with self.assertRaises(RuntimeError):
            env.step(4)

    def test_randomized_resets_differ(self):
        env = UCAVSimEnv(randomize=True, seed=42)
        a = env.reset()
        b = env.reset()
        self.assertFalse(np.allclose(a, b))

    def test_pursuit_converts_to_tail_position(self):
        """A scripted pursuit must end established in the bandit's rear
        hemisphere (equal speeds make actually closing to gun range from
        dead astern impossible, so the win itself is not required here)."""
        import math

        from dcs_bridge.geometry import heading_error, situation, wrap_heading

        env = UCAVSimEnv(max_steps=400, randomize=False, shaping=0.0)
        env.reset()
        done, info = False, {}
        while not done:
            dx = env.pos_b[0] - env.pos_r[0]
            dy = env.pos_b[1] - env.pos_r[1]
            bearing = wrap_heading(math.degrees(math.atan2(dx, dy)))
            e_psi = heading_error(bearing, env.act_r[2])
            want_climb = env.pos_r[2] < env.pos_b[2] + 300
            d_psi = 1 if e_psi > 5 else (-1 if e_psi < -5 else 0)
            d_gam = 1 if want_climb else (-1 if env.act_r[1] > 0 else 0)
            action = {(1, 1): 0, (1, 0): 1, (1, -1): 2,
                      (0, 1): 3, (0, 0): 4, (0, -1): 5,
                      (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}[(d_gam, d_psi)]
            _, _, done, info = env.step(action)

        feats = situation(env.pos_r, env.act_r, env.pos_b, env.act_b)
        self.assertLess(feats[0], 15.0, f"never pointed at the bandit: {info}")
        self.assertGreater(feats[1], 150.0, f"never got behind the bandit: {info}")

    def test_win_detected_from_firing_position(self):
        """Placed 2 km behind and 200 m above a fleeing bandit, one step wins."""
        env = UCAVSimEnv(max_steps=400, randomize=False, shaping=0.0)
        env.reset()
        env.pos_b = [130_000.0, 100_000.0, 3_000.0]
        env.act_b = [250.0, 0.0, 180.0]              # bandit running south
        env.pos_r = [130_000.0, 102_000.0, 3_200.0]  # we chase from the north, above
        env.act_r = [250.0, 0.0, 180.0]
        _, reward, done, info = env.step(4)          # hold everything
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "win")
        self.assertGreater(reward, 5.0)


if __name__ == "__main__":
    unittest.main()
