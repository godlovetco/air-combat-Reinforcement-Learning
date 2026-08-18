import os
import tempfile
import unittest

import numpy as np

from dcs_bridge import geometry as geo
from dcs_bridge.policy import QNetwork
from dcs_bridge.sim_env import UCAVSimEnv


class ActionSetTest(unittest.TestCase):
    def test_legacy_is_unchanged(self):
        cands = geo.candidate_actions(250.0, 0.0, 0.0)
        self.assertEqual(len(cands), 9)
        for v, _gamma, _psi in cands:
            self.assertEqual(v, 250.0)  # legacy holds speed exactly

    def test_energy_crosses_throttle_with_the_nine_maneuvers(self):
        cands = geo.candidate_actions(250.0, 0.0, 0.0, "energy")
        self.assertEqual(len(cands), 27)
        # Each consecutive triple shares a (gamma, psi) and differs in speed.
        for i in range(0, 27, 3):
            burner, hold, idle = cands[i], cands[i + 1], cands[i + 2]
            self.assertEqual(burner[1:], hold[1:])
            self.assertEqual(hold[1:], idle[1:])
            self.assertGreater(burner[0], hold[0])
            self.assertGreater(hold[0], idle[0])

    def test_unknown_action_set_rejected(self):
        with self.assertRaises(ValueError):
            geo.candidate_actions(250.0, 0.0, 0.0, "afterburner-only")
        with self.assertRaises(ValueError):
            geo.action_set_dims("afterburner-only")

    def test_dims_and_reverse_lookup(self):
        self.assertEqual(geo.action_set_dims("legacy"), (9, 72))
        self.assertEqual(geo.action_set_dims("energy"), (27, 216))
        self.assertEqual(geo.action_set_for(9), "legacy")
        self.assertEqual(geo.action_set_for(27), "energy")
        with self.assertRaises(ValueError):
            geo.action_set_for(13)


class EnergyModelTest(unittest.TestCase):
    def test_burner_accelerates_and_idle_decelerates_when_level(self):
        self.assertAlmostEqual(geo.energy_step(250.0, 0.0, 0.0, +1.0),
                               250.0 + geo.THROTTLE_ACCEL, places=6)
        self.assertAlmostEqual(geo.energy_step(250.0, 0.0, 0.0, -1.0),
                               250.0 - geo.THROTTLE_ACCEL, places=6)
        self.assertAlmostEqual(geo.energy_step(250.0, 0.0, 0.0, 0.0), 250.0, places=6)

    def test_climbing_costs_speed_and_diving_buys_it(self):
        climb = geo.energy_step(250.0, 30.0, 0.0, 0.0)
        dive = geo.energy_step(250.0, -30.0, 0.0, 0.0)
        self.assertLess(climb, 250.0)
        self.assertGreater(dive, 250.0)
        # Symmetric about level flight.
        self.assertAlmostEqual(250.0 - climb, dive - 250.0, places=6)

    def test_a_climb_at_burner_still_bleeds(self):
        # g*sin(30 deg) = 4.9 m/s^2 exceeds the 4.0 m/s^2 the engine buys.
        self.assertLess(geo.energy_step(250.0, 30.0, 0.0, +1.0), 250.0)

    def test_turning_bleeds_speed(self):
        straight = geo.energy_step(250.0, 0.0, 0.0, 0.0)
        turning = geo.energy_step(250.0, 0.0, 10.0, 0.0)
        self.assertLess(turning, straight)
        # Left and right cost the same.
        self.assertAlmostEqual(turning, geo.energy_step(250.0, 0.0, -10.0, 0.0),
                               places=6)

    def test_speed_stays_in_the_usable_band(self):
        self.assertEqual(geo.energy_step(geo.V_MAX, 0.0, 0.0, +1.0), geo.V_MAX)
        self.assertEqual(geo.energy_step(geo.V_MIN, 0.0, 0.0, -1.0), geo.V_MIN)

    def test_dt_scales_the_change(self):
        half = geo.energy_step(250.0, 0.0, 0.0, +1.0, dt=0.5)
        full = geo.energy_step(250.0, 0.0, 0.0, +1.0, dt=1.0)
        self.assertAlmostEqual(half - 250.0, (full - 250.0) / 2.0, places=6)


class NetworkShapeTest(unittest.TestCase):
    def test_network_input_width_follows_the_action_set(self):
        pos_r, act_r = [0.0, 0.0, 3000.0], [250.0, 0.0, 0.0]
        pos_b, act_b = [0.0, 5000.0, 3000.0], [250.0, 0.0, 180.0]
        self.assertEqual(
            geo.build_network_input(pos_r, act_r, pos_b, act_b).shape, (72,))
        self.assertEqual(
            geo.build_network_input(pos_r, act_r, pos_b, act_b,
                                    action_set="energy").shape, (216,))

    def test_energy_network_round_trips_and_reports_its_action_set(self):
        net = QNetwork(seed=1, input_dim=216, num_actions=27, hidden=(160, 60))
        self.assertEqual(net.action_set, "energy")
        self.assertEqual(net.forward(np.zeros(216)).shape, (27,))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "energy.npz")
            net.save(path)
            back = QNetwork.load(path)
        self.assertEqual((back.input_dim, back.hidden, back.num_actions),
                         (216, (160, 60), 27))
        for key, value in net.params.items():
            self.assertTrue(np.array_equal(back.params[key], value))

    def test_legacy_checkpoint_still_loads_unchanged(self):
        net = QNetwork.load("checkpoints/ucav_policy.npz")
        self.assertEqual((net.input_dim, net.num_actions), (72, 9))
        self.assertEqual(net.action_set, "legacy")

    def test_clone_preserves_shape(self):
        net = QNetwork(seed=2, input_dim=216, num_actions=27, hidden=(160, 60))
        twin = net.clone()
        self.assertEqual(twin.num_actions, 27)
        self.assertEqual(twin.hidden, (160, 60))

    def test_act_samples_within_the_action_range(self):
        net = QNetwork(seed=3, input_dim=216, num_actions=27, hidden=(16, 8))
        obs = np.zeros(216)
        for _ in range(200):
            self.assertIn(net.act(obs, epsilon=1.0), range(27))


class EnergyEnvTest(unittest.TestCase):
    def test_unknown_action_set_rejected(self):
        with self.assertRaises(ValueError):
            UCAVSimEnv(action_set="afterburner-only")

    def test_observation_and_action_space_match_the_action_set(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=1, action_set="energy")
        obs = env.reset()
        self.assertEqual(obs.shape, (216,))
        obs, _r, _done, _info = env.step(26)  # last action only exists here
        self.assertEqual(obs.shape, (216,))

    def test_agent_speed_actually_changes(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=2, action_set="energy")
        env.reset()
        v0 = env.act_r[0]
        for _ in range(5):
            env.step(3)  # hold climb, turn right, burner
        self.assertNotEqual(env.act_r[0], v0)

    def test_straight_bandit_keeps_constant_speed_in_energy_mode(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=3, action_set="energy")
        env.reset()
        v_b = env.act_b[0]
        for _ in range(5):
            env.step(4)
        self.assertEqual(env.act_b[0], v_b)

    def test_reactive_bandit_manages_energy(self):
        env = UCAVSimEnv(randomize=False, shaping=0.0, seed=4,
                         opponent="pursuit", action_set="energy")
        env.reset()
        v_b = env.act_b[0]
        for _ in range(5):
            env.step(4)
        self.assertNotEqual(env.act_b[0], v_b)
        self.assertGreaterEqual(env.act_b[0], geo.V_MIN)
        self.assertLessEqual(env.act_b[0], geo.V_MAX)

    def test_selfplay_uses_the_same_action_set(self):
        net = QNetwork(seed=5, input_dim=216, num_actions=27, hidden=(16, 8))
        env = UCAVSimEnv(randomize=True, shaping=0.05, seed=6, opponent="selfplay",
                         bandit_policy=net.clone(), action_set="energy")
        obs = env.reset()
        done, steps, info = False, 0, {}
        while not done and steps < env.max_steps + 1:
            obs, _r, done, info = env.step(net.act(obs))
            steps += 1
        self.assertTrue(done)
        self.assertIn(info["outcome"], ("win", "loss", "out_of_bounds", "timeout"))


class TrainWiringTest(unittest.TestCase):
    def test_parse_hidden(self):
        from dcs_bridge.policy import HIDDEN_SIZES
        from dcs_bridge.train import parse_hidden
        self.assertEqual(parse_hidden(None), HIDDEN_SIZES)
        self.assertEqual(parse_hidden("160,60"), (160, 60))
        for bad in ("160", "160,60,20", "a,b", "0,60"):
            with self.assertRaises(ValueError):
                parse_hidden(bad)

    def test_build_network_matches_the_action_set(self):
        from dcs_bridge.train import build_network, build_parser
        args = build_parser().parse_args(["--action-set", "energy", "--hidden", "160,60"])
        net = build_network(args)
        self.assertEqual((net.input_dim, net.num_actions), (216, 27))
        args = build_parser().parse_args([])
        self.assertEqual(build_network(args).num_actions, 9)

    def test_init_checkpoint_must_match_the_action_set(self):
        from dcs_bridge.train import build_network, build_parser
        args = build_parser().parse_args(
            ["--action-set", "energy", "--init", "checkpoints/ucav_policy.npz"]
        )
        with self.assertRaises(ValueError) as ctx:
            build_network(args)
        self.assertIn("action set", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
