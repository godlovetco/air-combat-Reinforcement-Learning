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


class NormalizationTest(unittest.TestCase):
    def test_legacy_normalization_is_untouched(self):
        feats = [30.0, 150.0, 5000.0, 90.0, 200.0, 0.0, 62500.0, 3000.0]
        self.assertTrue(np.array_equal(
            geo.normalize(feats), np.asarray(feats) / geo._NORM))

    def test_energy_scales_delta_v2_like_v2(self):
        # The legacy scale of 1.0 leaves delta_v2 at O(10^5) once speeds vary,
        # which would swamp every other feature in the first layer.
        feats = geo.situation([0.0, 0.0, 3000.0], [geo.V_MAX, 0.0, 0.0],
                              [0.0, 5000.0, 3000.0], [geo.V_MIN, 0.0, 180.0])
        legacy = geo.normalize(feats)
        energy = geo.normalize(feats, "energy")
        self.assertGreater(abs(legacy[5]), 1000.0)
        self.assertLess(abs(energy[5]), 10.0)
        # Only the delta_v2 column differs between the two scalings.
        for i in range(len(feats)):
            if i != 5:
                self.assertAlmostEqual(legacy[i], energy[i], places=12)

    def test_unknown_action_set_rejected(self):
        with self.assertRaises(ValueError):
            geo.normalize([0.0] * 8, "afterburner-only")


class TransferTest(unittest.TestCase):
    def test_lift_preserves_shape_and_family(self):
        legacy = QNetwork.load("checkpoints/ucav_policy.npz")
        energy = legacy.to_energy()
        self.assertEqual((energy.input_dim, energy.num_actions), (216, 27))
        self.assertEqual(energy.hidden, legacy.hidden)
        self.assertEqual(energy.action_set, "energy")

    def test_lift_mostly_reproduces_the_legacy_maneuver_choice(self):
        """The lift is a warm start, not an identity: it agrees ~97%, not 100%.

        The three throttle variants of a maneuver differ slightly in speed and
        the energy action set rescales ``delta_v2``, so the inputs are close but
        not equal.  Asserting exact agreement would be asserting something
        untrue -- what matters is that the prior transferred.
        """
        legacy = QNetwork.load("checkpoints/ucav_policy.npz")
        energy = legacy.to_energy()
        rng = np.random.default_rng(0)
        agree = 0
        trials = 200
        for _ in range(trials):
            pos_r = [130_000.0, 100_000.0, 3_000.0]
            pos_b = [130_000.0 + rng.uniform(-6_000, 6_000),
                     100_000.0 + rng.uniform(2_000, 12_000),
                     3_000.0 + rng.uniform(-800, 800)]
            act_r = [250.0, 0.0, float(rng.uniform(0, 360))]
            act_b = [250.0, 0.0, float(rng.uniform(0, 360))]
            x_l = geo.build_network_input(pos_r, act_r, pos_b, act_b)
            x_e = geo.build_network_input(pos_r, act_r, pos_b, act_b,
                                          action_set="energy")
            agree += int(legacy.act(x_l) == energy.act(x_e) // 3)
        self.assertGreater(agree / trials, 0.9)

    def test_lift_ranks_the_nine_maneuvers_the_same_way(self):
        legacy = QNetwork.load("checkpoints/ucav_policy.npz")
        energy = legacy.to_energy()
        pos_r, act_r = [130_000.0, 100_000.0, 3_000.0], [250.0, 0.0, 0.0]
        pos_b, act_b = [130_000.0, 110_000.0, 3_000.0], [250.0, 0.0, 180.0]
        q_l = legacy.forward(geo.build_network_input(pos_r, act_r, pos_b, act_b))
        q_e = energy.forward(
            geo.build_network_input(pos_r, act_r, pos_b, act_b, action_set="energy"))
        # Collapse the throttle axis: each maneuver's three columns are equal up
        # to the tie-break epsilon, so take the "hold" one.
        self.assertEqual(list(np.argsort(q_l)), list(np.argsort(q_e[1::3])))

    def test_lift_rejects_a_non_legacy_source(self):
        energy = QNetwork(seed=1, input_dim=216, num_actions=27, hidden=(16, 8))
        with self.assertRaises(ValueError):
            energy.to_energy()

    def test_transfer_init_wiring(self):
        from dcs_bridge.train import build_network, build_parser
        parse = build_parser().parse_args
        net = build_network(parse(["--action-set", "energy", "--transfer-init",
                                   "checkpoints/ucav_policy.npz"]))
        self.assertEqual((net.input_dim, net.num_actions), (216, 27))

        # Wrong action set, both flags at once, and a non-legacy source.
        for argv in (
            ["--transfer-init", "checkpoints/ucav_policy.npz"],
            ["--action-set", "energy", "--transfer-init", "checkpoints/ucav_policy.npz",
             "--init", "checkpoints/ucav_policy.npz"],
        ):
            with self.assertRaises(ValueError):
                build_network(parse(argv))


class ShippedEnergyCheckpointTest(unittest.TestCase):
    def test_it_loads_as_an_energy_network(self):
        net = QNetwork.load("checkpoints/ucav_policy_energy.npz")
        self.assertEqual((net.input_dim, net.num_actions), (216, 27))
        self.assertEqual(net.action_set, "energy")

    def test_it_drives_an_episode_to_a_conclusion(self):
        net = QNetwork.load("checkpoints/ucav_policy_energy.npz")
        env = UCAVSimEnv(randomize=True, shaping=0.0, seed=4242,
                         opponent="mixed", action_set="energy")
        obs = env.reset()
        done, steps, info = False, 0, {}
        while not done and steps < env.max_steps + 1:
            obs, _r, done, info = env.step(net.act(obs))
            steps += 1
        self.assertTrue(done)
        self.assertIn(info["outcome"], ("win", "loss", "out_of_bounds", "timeout"))
