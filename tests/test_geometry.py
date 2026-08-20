import math
import unittest

import numpy as np

from dcs_bridge import geometry as geo


class GeometryTest(unittest.TestCase):
    def test_matches_legacy_class_env(self):
        """q_r, d, beta, delta_h must equal the legacy generate_state values.

        (q_b is excluded: the legacy code forgot the degree->radian
        conversion there, which this package fixes.)
        """
        import class_env

        legacy = class_env.AirCombat()
        pos_r, pos_b = [130000, 100000, 3000], [131000, 108000, 3500]
        act_r, act_b = [250, 10, 30], [250, 0, 180]
        expected = legacy.generate_state(pos_r, pos_b, act_r, act_b)
        got = geo.situation(pos_r, act_r, pos_b, act_b)

        for idx in (0, 2, 3, 4, 5, 6, 7):  # everything except q_b
            self.assertAlmostEqual(got[idx], expected[idx], places=6,
                                   msg=f"feature {idx} diverges from legacy")

    def test_q_b_head_on(self):
        # Bandit flying due south straight at us from the north: aspect 0.
        feats = geo.situation(
            [0, 0, 3000], [250, 0, 0],
            [0, 10000, 3000], [250, 0, 180],
        )
        self.assertAlmostEqual(feats[0], 0.0, places=6)  # q_r: nose-on
        self.assertAlmostEqual(feats[1], 0.0, places=6)  # q_b: also nose-on
        self.assertAlmostEqual(feats[2], 10000.0, places=6)
        self.assertAlmostEqual(feats[3], 180.0, places=6)  # opposite headings

    def test_q_b_tail_chase(self):
        # Bandit flying away from us: its aspect to us is 180.
        feats = geo.situation(
            [0, 0, 3000], [250, 0, 0],
            [0, 5000, 3000], [250, 0, 0],
        )
        self.assertAlmostEqual(feats[1], 180.0, places=6)

    def test_candidate_actions_shape_and_order(self):
        cands = geo.candidate_actions(250, 0, 0)
        self.assertEqual(len(cands), 9)
        self.assertEqual(cands[4], [250, 0.0, 0.0])       # hold everything
        self.assertEqual(cands[0][1], 10.0)               # climb+ turn right
        self.assertEqual(cands[0][2], 10.0)
        self.assertEqual(cands[8][1], -10.0)              # descend turn left
        self.assertEqual(cands[8][2], 350.0)              # wrapped

    def test_gamma_clamped(self):
        cands = geo.candidate_actions(250, geo.GAMMA_LIMIT_DEG, 0)
        self.assertEqual(cands[0][1], geo.GAMMA_LIMIT_DEG)

    def test_network_input_shape(self):
        x = geo.build_network_input(
            [0, 0, 3000], [250, 0, 0], [0, 10000, 3000], [250, 0, 180]
        )
        self.assertEqual(x.shape, (geo.INPUT_DIM,))
        self.assertTrue(np.isfinite(x).all())

    def test_network_input_bandit_override(self):
        base = geo.build_network_input(
            [0, 0, 3000], [250, 0, 0], [0, 10000, 3000], [250, 0, 180]
        )
        moved = geo.build_network_input(
            [0, 0, 3000], [250, 0, 0], [0, 10000, 3000], [250, 0, 180],
            next_pos_b=[3000, 9000, 3200],
        )
        self.assertFalse(np.allclose(base, moved))

    def test_heading_error(self):
        self.assertAlmostEqual(geo.heading_error(10, 350), 20.0)
        self.assertAlmostEqual(geo.heading_error(350, 10), -20.0)
        self.assertAlmostEqual(geo.heading_error(180, 0), -180.0)

    def test_point_mass_speed(self):
        nxt = geo.step_point_mass([0, 0, 1000], [250, 20, 77], dt=1.0)
        dist = math.dist(nxt, [0, 0, 1000])
        self.assertAlmostEqual(dist, 250.0, places=6)


if __name__ == "__main__":
    unittest.main()
