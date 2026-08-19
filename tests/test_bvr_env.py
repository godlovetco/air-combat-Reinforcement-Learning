import math
import unittest

import numpy as np

from dcs_bridge import bvr
from dcs_bridge import geometry as geo
from dcs_bridge.bvr_env import (ARENA_XY, BVR_STATE_DIM, MERGE_RANGE, OUTCOMES,
                                START_RANGE, BVRSimEnv, observation_dim)
from dcs_bridge.policy import QNetwork


def level_action(env, want_psi):
    """Index of the candidate closest to a heading, flattest climb first."""
    cands = geo.candidate_actions(*env.act_r, action_set=env.action_set, dt=env.dt)
    return min(range(len(cands)),
               key=lambda i: (abs(geo.heading_error(want_psi, cands[i][2])),
                              abs(cands[i][1])))


class ConstructionTest(unittest.TestCase):
    def test_observation_dim(self):
        self.assertEqual(observation_dim("legacy"), 9 * 8 + BVR_STATE_DIM)
        self.assertEqual(observation_dim("energy"), 27 * 8 + BVR_STATE_DIM)

    def test_observation_matches_the_declared_width(self):
        for action_set in ("legacy", "energy"):
            env = BVRSimEnv(seed=1, action_set=action_set)
            self.assertEqual(env.reset().shape, (observation_dim(action_set),))

    def test_rejects_bad_arguments(self):
        for kwargs in (
            {"action_set": "afterburner-only"},
            {"opponent": "kamikaze"},
            {"opponent": "mixed", "mixed_weights": {"kamikaze": 1}},
            {"opponent": "mixed", "mixed_weights": {"pursuit": 0.0}},
            {"opponent": "selfplay"},          # no bandit policy
            {"missiles": 0},
        ):
            with self.assertRaises(ValueError):
                BVRSimEnv(seed=1, **kwargs)

    def test_starts_at_bvr_range_head_on(self):
        env = BVRSimEnv(seed=2, randomize=False)
        env.reset()
        _los, d = bvr.line_of_sight(env.pos_r, env.pos_b)
        self.assertAlmostEqual(d, START_RANGE, delta=1.0)
        self.assertEqual(env.missiles_r, env.loadout)
        self.assertEqual(env.missiles_b, env.loadout)
        self.assertEqual(env.in_flight, [])

    def test_randomized_starts_vary_but_stay_beyond_visual_range(self):
        env = BVRSimEnv(seed=3, randomize=True)
        seen = set()
        for _ in range(20):
            env.reset()
            _los, d = bvr.line_of_sight(env.pos_r, env.pos_b)
            self.assertGreater(d, MERGE_RANGE * 3)
            seen.add(round(d))
        self.assertGreater(len(seen), 1)


class WeaponsTest(unittest.TestCase):
    def setUp(self):
        self.env = BVRSimEnv(seed=4, randomize=False, shaping=0.0,
                             opponent="straight")
        self.env.reset()

    def test_no_shot_without_a_lock(self):
        # Put the bandit far off the nose: outside the radar gimbal.
        self.env.act_r = [250.0, 0.0, 90.0]
        self.env._update_tracks()
        self.assertFalse(self.env._lock("agent"))
        self.assertFalse(self.env._try_launch("agent"))
        self.assertEqual(self.env.missiles_r, self.env.loadout)

    def test_no_shot_at_a_notching_target(self):
        self.env.act_b = [250.0, 0.0, 90.0]   # bandit beaming us
        self.env._update_tracks()
        self.assertFalse(self.env._lock("agent"))
        self.assertFalse(self.env._try_launch("agent"))

    def test_shot_inside_the_envelope(self):
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 30_000.0, 8000.0]
        self.env.act_b = [250.0, 0.0, 180.0]
        self.assertTrue(self.env._try_launch("agent"))
        self.assertEqual(self.env.missiles_r, self.env.loadout - 1)
        self.assertEqual(len(self.env.in_flight), 1)

    def test_trigger_discipline_blocks_an_immediate_second_shot(self):
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 20_000.0, 8000.0]
        self.env.act_b = [250.0, 0.0, 180.0]
        self.assertTrue(self.env._try_launch("agent"))
        self.assertFalse(self.env._try_launch("agent"))

    def test_no_second_shot_outside_the_nez_while_one_is_unsupported(self):
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 55_000.0, 8000.0]
        self.env.act_b = [250.0, 0.0, 180.0]
        self.assertTrue(self.env._try_launch("agent"))     # maddog, outside NEZ
        self.env._last_shot["agent"] = -1e9                # ignore the cooldown
        self.assertFalse(self.env._try_launch("agent"))    # still supporting it

    def test_winchester(self):
        env = BVRSimEnv(seed=5, randomize=False, shaping=0.0, missiles=1)
        env.reset()
        env.pos_b = [env.pos_r[0], env.pos_r[1] + 20_000.0, 8000.0]
        env.act_b = [250.0, 0.0, 180.0]
        self.assertTrue(env._try_launch("agent"))
        env._last_shot["agent"] = -1e9
        self.assertFalse(env._try_launch("agent"))
        self.assertEqual(env.missiles_r, 0)


class OutcomeTest(unittest.TestCase):
    def test_a_hit_on_the_bandit_is_a_win(self):
        env = BVRSimEnv(seed=6, randomize=False, shaping=0.0, opponent="straight")
        env.reset()
        env.in_flight.append(bvr.Missile.launch(env.spec, "agent",
                                                env.pos_b, env.act_b))
        _obs, reward, done, info = env.step(4)
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "win")
        self.assertGreater(reward, 5.0)

    def test_a_hit_on_us_is_a_loss(self):
        env = BVRSimEnv(seed=7, randomize=False, shaping=0.0, opponent="straight")
        env.reset()
        env.in_flight.append(bvr.Missile.launch(env.spec, "bandit",
                                                env.pos_r, env.act_r))
        _obs, reward, done, info = env.step(4)
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "loss")
        self.assertLess(reward, -5.0)

    def test_both_winchester_inside_visual_range_is_a_merge(self):
        env = BVRSimEnv(seed=8, randomize=False, shaping=0.0, opponent="straight")
        env.reset()
        env.missiles_r = env.missiles_b = 0
        env.pos_b = [env.pos_r[0], env.pos_r[1] + 3_000.0, 8000.0]
        _obs, _reward, done, info = env.step(4)
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "merge")

    def test_departure_is_attributed(self):
        env = BVRSimEnv(seed=9, randomize=False, shaping=0.0, opponent="straight")
        env.reset()
        env.pos_b = [-10.0, 100.0, 8000.0]
        _obs, reward, done, info = env.step(4)
        self.assertEqual(info["outcome"], "bandit_departed")
        self.assertEqual(reward, 0.0)

        env = BVRSimEnv(seed=10, randomize=False, shaping=0.0, opponent="straight")
        env.reset()
        env.pos_r = [-10.0, 100.0, 8000.0]
        _obs, reward, done, info = env.step(4)
        self.assertEqual(info["outcome"], "out_of_bounds")
        self.assertEqual(reward, -5.0)

    def test_every_episode_reaches_a_known_outcome(self):
        net = QNetwork(seed=11, input_dim=observation_dim("legacy"), num_actions=9)
        for opponent in ("straight", "pursuit", "evasive", "ace"):
            env = BVRSimEnv(seed=12, opponent=opponent)
            obs = env.reset()
            done, info = False, {}
            while not done:
                obs, _r, done, info = env.step(net.act(obs))
            self.assertIn(info["outcome"], OUTCOMES)


class ObservationBlockTest(unittest.TestCase):
    def setUp(self):
        self.env = BVRSimEnv(seed=13, randomize=False, shaping=0.0,
                             opponent="straight")
        self.env.reset()

    def block(self):
        return self.env._bvr_block(self.env.pos_r, self.env.act_r,
                                   self.env.pos_b, self.env.act_b, "agent")

    def test_width_and_range(self):
        b = self.block()
        self.assertEqual(b.shape, (BVR_STATE_DIM,))
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(b >= -1.0) and np.all(b <= 1.0))

    def test_lock_and_spike_flags(self):
        self.assertEqual(self.block()[0], 1.0)   # we see them
        self.assertEqual(self.block()[1], 1.0)   # and they see us
        self.env.act_b = [250.0, 0.0, 90.0]      # bandit beams: we lose it
        self.env._update_tracks()
        self.assertEqual(self.block()[0], 0.0)

    def test_missile_counts_are_reported(self):
        self.env.missiles_r = 1
        self.assertAlmostEqual(self.block()[2], 1.0 / self.env.loadout)

    def test_envelope_flags_turn_on_as_the_range_closes(self):
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 80_000.0, 8000.0]
        self.env.act_b = [250.0, 0.0, 180.0]
        self.assertEqual(self.block()[9], 0.0)   # out of the envelope at 80 km
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 12_000.0, 8000.0]
        b = self.block()
        self.assertEqual(b[9], 1.0)              # in the envelope
        self.assertEqual(b[10], 1.0)             # and in the no-escape zone


class ShapingTest(unittest.TestCase):
    def setUp(self):
        self.env = BVRSimEnv(seed=14, randomize=False, shaping=0.05,
                             opponent="straight")
        self.env.reset()

    def test_neutral_with_nothing_in_the_air(self):
        self.assertEqual(self.env._shaping_term(), 0.0)

    def test_holding_the_lock_pays_while_supporting_a_shot(self):
        self.env.pos_b = [self.env.pos_r[0], self.env.pos_r[1] + 40_000.0, 8000.0]
        self.env.act_b = [250.0, 0.0, 180.0]
        self.env.in_flight.append(bvr.Missile.launch(self.env.spec, "agent",
                                                     self.env.pos_r, self.env.act_r))
        self.env._update_tracks()
        self.assertEqual(self.env._shaping_term(), 1.0)
        self.env.act_r = [250.0, 0.0, 120.0]   # crank past the gimbal: lock lost
        self.env._update_tracks()
        self.assertEqual(self.env._shaping_term(), -1.0)

    def test_being_tracked_costs_and_notching_pays(self):
        threat = bvr.Missile.launch(self.env.spec, "bandit",
                                    [self.env.pos_r[0], self.env.pos_r[1] + 8_000.0,
                                     8000.0], [900.0, 0.0, 180.0])
        threat.active = True
        self.env.in_flight.append(threat)
        self.assertEqual(self.env._shaping_term(), -1.0)
        self.env.act_r = [250.0, 0.0, 90.0]    # beam it
        self.assertEqual(self.env._shaping_term(), 1.0)


class BanditTest(unittest.TestCase):
    def _env(self, opponent):
        env = BVRSimEnv(seed=15, randomize=False, shaping=0.0, opponent=opponent)
        env.reset()
        return env

    def test_pursuit_points_at_us(self):
        env = self._env("pursuit")
        env.act_b = [250.0, 0.0, 90.0]
        before = abs(geo.heading_error(180.0, env.act_b[2]))
        env.step(4)
        self.assertLess(abs(geo.heading_error(180.0, env.act_b[2])), before)

    def test_ace_notches_an_incoming_missile(self):
        env = self._env("ace")
        threat = bvr.Missile.launch(env.spec, "agent", env.pos_r, env.act_r)
        threat.active = True
        env.in_flight.append(threat)
        env.step(4)
        # Turning toward the beam means turning away from pointing at us (180).
        self.assertGreater(abs(geo.heading_error(180.0, env.act_b[2])), 0.0)

    def test_ace_cranks_while_supporting_its_own_shot(self):
        env = self._env("ace")
        env.in_flight.append(bvr.Missile.launch(env.spec, "bandit",
                                                env.pos_b, env.act_b))
        env.step(4)
        off = abs(geo.heading_error(180.0, env.act_b[2]))
        self.assertGreater(off, 0.0)      # it turned off pure pursuit
        self.assertLess(off, 90.0)        # but not all the way to the beam

    def test_selfplay_bandit_sees_a_bvr_observation(self):
        seen = {}

        class Recorder:
            def act(self, obs, epsilon=0.0):
                seen["dim"] = obs.shape
                return 4

        env = BVRSimEnv(seed=16, randomize=False, shaping=0.0,
                        opponent="selfplay", bandit_policy=Recorder())
        env.reset()
        env.step(4)
        self.assertEqual(seen["dim"], (observation_dim("legacy"),))


if __name__ == "__main__":
    unittest.main()


class RadarIntegrationTest(unittest.TestCase):
    def _env(self, **kw):
        env = BVRSimEnv(seed=20, randomize=False, shaping=0.0,
                        opponent="straight", **kw)
        env.reset()
        return env

    def test_both_sides_start_the_merge_already_tracking(self):
        env = self._env()
        self.assertTrue(env._lock("agent"))
        self.assertTrue(env._lock("bandit"))

    def test_a_lost_track_is_dropped_at_once(self):
        env = self._env()
        env.act_b = [250.0, 0.0, 90.0]      # bandit notches us
        env._update_tracks()
        self.assertFalse(env._lock("agent"))

    def test_a_mechanical_set_is_slow_to_re_acquire(self):
        env = self._env()
        env.act_b = [250.0, 0.0, 90.0]
        env._update_tracks()
        env.act_b = [250.0, 0.0, 180.0]     # bandit comes back hot
        env._update_tracks()
        self.assertFalse(env._lock("agent"))   # 1 s of a 3 s re-acquisition
        env._update_tracks()
        env._update_tracks()
        self.assertTrue(env._lock("agent"))

    def test_an_aesa_is_back_almost_immediately(self):
        env = self._env(radar=bvr.Radar.aesa())
        env.act_b = [250.0, 0.0, 90.0]
        env._update_tracks()
        self.assertFalse(env._lock("agent"))
        env.act_b = [250.0, 0.0, 180.0]
        env._update_tracks()
        self.assertTrue(env._lock("agent"))

    def test_lpi_hides_the_lock_from_the_rwr(self):
        # Bandit carries an AESA; at 70 km it is tracking us without warning us.
        env = self._env(bandit_radar=bvr.Radar.aesa())
        self.assertTrue(env._lock("bandit"))
        self.assertFalse(env._spiked("agent"))
        mech = self._env()
        self.assertTrue(mech._lock("bandit"))
        self.assertTrue(mech._spiked("agent"))

    def test_a_mechanical_radar_can_only_support_one_shot(self):
        env = self._env()
        env.pos_b = [env.pos_r[0], env.pos_r[1] + 20_000.0, 8000.0]
        env.act_b = [250.0, 0.0, 180.0]
        env._update_tracks()
        self.assertTrue(env._try_launch("agent"))
        env._last_shot["agent"] = -1e9
        self.assertFalse(env._try_launch("agent"))   # still guiding the first

    def test_an_aesa_can_support_several(self):
        env = self._env(radar=bvr.Radar.aesa())
        env.pos_b = [env.pos_r[0], env.pos_r[1] + 20_000.0, 8000.0]
        env.act_b = [250.0, 0.0, 180.0]
        env._update_tracks()
        for _ in range(3):
            env._last_shot["agent"] = -1e9
            self.assertTrue(env._try_launch("agent"))
        self.assertEqual(len(env._shots("agent")), 3)

    def test_asymmetric_loadouts(self):
        env = self._env(spec=bvr.ARH_LONG, bandit_spec=bvr.IR_SHORT)
        self.assertIs(env.spec_of("agent"), bvr.ARH_LONG)
        self.assertIs(env.spec_of("bandit"), bvr.IR_SHORT)
        env.pos_b = [env.pos_r[0], env.pos_r[1] + 40_000.0, 8000.0]
        env.act_b = [250.0, 0.0, 180.0]
        env._update_tracks()
        self.assertTrue(env._try_launch("agent"))    # ramjet reaches
        self.assertFalse(env._try_launch("bandit"))  # 40 km is far outside IR range


class TrainWiringTest(unittest.TestCase):
    def test_network_dims_per_engagement(self):
        from dcs_bridge.train import network_dims
        self.assertEqual(network_dims("legacy", "wvr"), (9, 72))
        self.assertEqual(network_dims("legacy", "bvr"), (9, 72 + BVR_STATE_DIM))
        self.assertEqual(network_dims("energy", "bvr"), (27, 216 + BVR_STATE_DIM))
        with self.assertRaises(ValueError):
            network_dims("legacy", "within-visual-range")

    def test_make_env_picks_the_right_environment(self):
        from dcs_bridge.sim_env import UCAVSimEnv
        from dcs_bridge.train import make_env
        self.assertIsInstance(make_env("bvr", seed=1), BVRSimEnv)
        self.assertIsInstance(make_env("wvr", seed=1), UCAVSimEnv)

    def test_build_network_sizes_the_bvr_input(self):
        from dcs_bridge.train import build_network, build_parser
        net = build_network(build_parser().parse_args(["--engagement", "bvr"]))
        self.assertEqual(net.input_dim, observation_dim("legacy"))
        self.assertEqual(net.num_actions, 9)

    def test_init_from_a_wvr_checkpoint_is_refused(self):
        from dcs_bridge.train import build_network, build_parser
        args = build_parser().parse_args(
            ["--engagement", "bvr", "--init", "checkpoints/ucav_policy.npz"])
        with self.assertRaises(ValueError) as ctx:
            build_network(args)
        self.assertIn("input", str(ctx.exception))

    def test_transfer_init_is_refused_for_bvr(self):
        from dcs_bridge.train import build_network, build_parser
        args = build_parser().parse_args(
            ["--engagement", "bvr", "--action-set", "energy",
             "--transfer-init", "checkpoints/ucav_policy.npz"])
        with self.assertRaises(ValueError):
            build_network(args)

    def test_default_episode_length_follows_the_engagement(self):
        from dcs_bridge.train import build_parser, resolve_max_steps
        parse = build_parser().parse_args
        self.assertEqual(resolve_max_steps(parse([])), 400)
        self.assertEqual(resolve_max_steps(parse(["--engagement", "bvr"])), 600)
        self.assertEqual(
            resolve_max_steps(parse(["--engagement", "bvr", "--max-steps", "900"])),
            900)

    def test_evaluate_reports_survival_for_bvr(self):
        from dcs_bridge.train import evaluate
        net = QNetwork(seed=30, input_dim=observation_dim("legacy"), num_actions=9)
        win, survival = evaluate(net, episodes=6, seed=31, opponent="ace",
                                 engagement="bvr")
        self.assertGreaterEqual(survival, win)   # every win is also a survival
        self.assertLessEqual(survival, 1.0)
