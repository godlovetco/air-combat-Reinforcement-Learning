import math
import unittest

from dcs_bridge import bvr
from dcs_bridge import geometry as geo


NORTH = 0.0
SOUTH = 180.0
EAST = 90.0


def fly(pos, act, dt=1.0):
    return geo.step_point_mass(pos, act, dt)


class GeometryHelpersTest(unittest.TestCase):
    def test_line_of_sight(self):
        los, d = bvr.line_of_sight([0.0, 0.0, 0.0], [0.0, 1000.0, 0.0])
        self.assertAlmostEqual(d, 1000.0)
        self.assertAlmostEqual(los[1], 1.0)

    def test_coincident_points_are_safe(self):
        los, d = bvr.line_of_sight([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(d, 0.0)
        self.assertEqual(los, (0.0, 0.0, 0.0))

    def test_closure_is_positive_head_on(self):
        c = bvr.closure_rate([0.0, 0.0, 8000.0], [250.0, 0.0, NORTH],
                             [0.0, 40_000.0, 8000.0], [250.0, 0.0, SOUTH])
        self.assertAlmostEqual(c, 500.0, places=3)

    def test_closure_is_negative_when_the_target_outruns_us(self):
        c = bvr.closure_rate([0.0, 0.0, 8000.0], [250.0, 0.0, NORTH],
                             [0.0, 40_000.0, 8000.0], [300.0, 0.0, NORTH])
        self.assertAlmostEqual(c, -50.0, places=3)

    def test_radial_speed_ignores_the_observer(self):
        # Same target motion seen by a fast and a stationary observer.
        tgt, tgt_act = [0.0, 40_000.0, 8000.0], [250.0, 0.0, SOUTH]
        a = bvr.radial_speed([0.0, 0.0, 8000.0], tgt, tgt_act)
        self.assertAlmostEqual(a, -250.0, places=3)  # closing on us

    def test_radial_speed_is_zero_when_beaming(self):
        r = bvr.radial_speed([0.0, 0.0, 8000.0], [0.0, 40_000.0, 8000.0],
                             [250.0, 0.0, EAST])
        self.assertAlmostEqual(r, 0.0, places=6)

    def test_off_boresight(self):
        own_pos, own_act = [0.0, 0.0, 8000.0], [250.0, 0.0, NORTH]
        self.assertAlmostEqual(
            bvr.off_boresight(own_pos, own_act, [0.0, 40_000.0, 8000.0]), 0.0,
            places=6)
        self.assertAlmostEqual(
            bvr.off_boresight(own_pos, own_act, [40_000.0, 0.0, 8000.0]), 90.0,
            places=6)


class RadarTest(unittest.TestCase):
    def setUp(self):
        self.radar = bvr.Radar()
        self.own_pos = [0.0, 0.0, 8000.0]
        self.own_act = [250.0, 0.0, NORTH]

    def test_sees_a_closing_target_on_the_nose(self):
        self.assertTrue(self.radar.can_see(
            self.own_pos, self.own_act, [0.0, 40_000.0, 8000.0], [250.0, 0.0, SOUTH]))

    def test_loses_a_beaming_target_to_the_notch(self):
        tgt, tgt_act = [0.0, 40_000.0, 8000.0], [250.0, 0.0, EAST]
        self.assertFalse(self.radar.can_see(self.own_pos, self.own_act, tgt, tgt_act))
        self.assertTrue(self.radar.in_notch(self.own_pos, self.own_act, tgt, tgt_act))

    def test_the_notch_does_not_depend_on_our_own_speed(self):
        """A fast shooter closing hard still loses a target that beams it."""
        tgt, tgt_act = [0.0, 40_000.0, 8000.0], [250.0, 0.0, EAST]
        for own_speed in (150.0, 250.0, 400.0):
            self.assertFalse(self.radar.can_see(
                self.own_pos, [own_speed, 0.0, NORTH], tgt, tgt_act))

    def test_loses_a_target_outside_the_gimbal_limit(self):
        tgt, tgt_act = [40_000.0, 0.0, 8000.0], [250.0, 0.0, 270.0]
        self.assertFalse(self.radar.can_see(self.own_pos, self.own_act, tgt, tgt_act))
        self.assertFalse(self.radar.in_notch(self.own_pos, self.own_act, tgt, tgt_act))

    def test_loses_a_target_beyond_detection_range(self):
        self.assertFalse(self.radar.can_see(
            self.own_pos, self.own_act,
            [0.0, bvr.RADAR_RANGE + 1000.0, 8000.0], [250.0, 0.0, SOUTH]))

    def test_a_target_just_outside_the_notch_is_held(self):
        # Nudge the beaming target so it has a little radial velocity.
        tgt = [0.0, 40_000.0, 8000.0]
        just_out = geo.wrap_heading(EAST + 20.0)
        self.assertGreater(abs(bvr.radial_speed(self.own_pos, tgt,
                                                [250.0, 0.0, just_out])),
                           bvr.NOTCH_CLOSURE)
        self.assertTrue(self.radar.can_see(self.own_pos, self.own_act, tgt,
                                           [250.0, 0.0, just_out]))


class EnvelopeTest(unittest.TestCase):
    def test_altitude_buys_range(self):
        self.assertAlmostEqual(bvr.altitude_factor(0.0), 0.6)
        self.assertAlmostEqual(bvr.altitude_factor(10_000.0), 1.0)
        self.assertAlmostEqual(bvr.altitude_factor(20_000.0), 1.0)  # clamped
        self.assertLess(bvr.altitude_factor(3_000.0), bvr.altitude_factor(9_000.0))

    def test_aspect_factor_is_worst_against_a_runner(self):
        self.assertAlmostEqual(bvr.aspect_factor(0.0), 1.0)
        self.assertAlmostEqual(bvr.aspect_factor(180.0), 0.35)
        self.assertGreater(bvr.aspect_factor(0.0), bvr.aspect_factor(90.0))
        self.assertGreater(bvr.aspect_factor(90.0), bvr.aspect_factor(180.0))

    def test_envelope_shrinks_from_head_on_to_dragging(self):
        own_pos, own_act = [0.0, 0.0, 8000.0], [250.0, 0.0, NORTH]
        tgt = [0.0, 40_000.0, 8000.0]
        reach = []
        for heading in (SOUTH, EAST, NORTH):  # closing, beaming, running
            la = bvr.launch_authority(bvr.DEFAULT_MISSILE, own_pos, own_act, tgt,
                                      [250.0, 0.0, heading])
            reach.append(la["rmax"])
            self.assertGreater(la["rmax"], la["rne"])
        self.assertGreater(reach[0], reach[1])
        self.assertGreater(reach[1], reach[2])

    def test_a_shot_inside_the_nez_is_also_inside_the_envelope(self):
        own_pos, own_act = [0.0, 0.0, 9000.0], [250.0, 0.0, NORTH]
        la = bvr.launch_authority(bvr.DEFAULT_MISSILE, own_pos, own_act,
                                  [0.0, 15_000.0, 9000.0], [250.0, 0.0, SOUTH])
        self.assertTrue(la["in_nez"])
        self.assertTrue(la["in_envelope"])

    def test_too_close_to_arm(self):
        own_pos, own_act = [0.0, 0.0, 9000.0], [250.0, 0.0, NORTH]
        la = bvr.launch_authority(bvr.DEFAULT_MISSILE, own_pos, own_act,
                                  [0.0, 800.0, 9000.0], [250.0, 0.0, SOUTH])
        self.assertFalse(la["in_envelope"])
        self.assertFalse(la["in_nez"])


class MissileTest(unittest.TestCase):
    def setUp(self):
        self.shooter_pos = [0.0, 0.0, 9000.0]
        self.shooter_act = [250.0, 0.0, NORTH]

    def _launch(self):
        return bvr.Missile.launch(bvr.DEFAULT_MISSILE, "agent",
                                  self.shooter_pos, self.shooter_act)

    def test_it_hits_a_target_that_does_nothing(self):
        m = self._launch()
        tgt, tgt_act = [0.0, 30_000.0, 9000.0], [250.0, 0.0, SOUTH]
        outcome = None
        for _ in range(200):
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "hit")

    def test_dropping_the_lock_before_pitbull_throws_the_shot_away(self):
        m = self._launch()
        tgt, tgt_act = [0.0, 60_000.0, 9000.0], [250.0, 0.0, SOUTH]
        outcome = None
        for _ in range(200):
            outcome = m.step(tgt, tgt_act, supported=False)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "no_lock")
        self.assertFalse(m.active)

    def test_dropping_the_lock_after_pitbull_does_not(self):
        m = self._launch()
        tgt, tgt_act = [0.0, 30_000.0, 9000.0], [250.0, 0.0, SOUTH]
        outcome = None
        for _ in range(200):
            # Support only until the seeker takes over, then stop.
            outcome = m.step(tgt, tgt_act, supported=not m.active)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertTrue(m.active)
        self.assertEqual(outcome, "hit")

    def test_beaming_an_active_missile_defeats_it(self):
        m = self._launch()
        tgt = [0.0, 30_000.0, 9000.0]
        outcome = None
        for _ in range(200):
            # Run at the missile until it goes active, then break to the beam.
            tgt_act = [250.0, 0.0, EAST if m.active else SOUTH]
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "notched")

    def test_it_runs_out_of_energy_against_a_long_drag(self):
        spec = bvr.MissileSpec(max_flight_time=8.0)
        m = bvr.Missile.launch(spec, "agent", self.shooter_pos, self.shooter_act)
        tgt, tgt_act = [0.0, 80_000.0, 9000.0], [300.0, 0.0, NORTH]
        outcome = None
        for _ in range(200):
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "out_of_energy")

    def test_support_owed_falls_to_zero_at_pitbull(self):
        m = self._launch()
        tgt, tgt_act = [0.0, 40_000.0, 9000.0], [250.0, 0.0, SOUTH]
        first = m.time_to_active(tgt)
        self.assertGreater(first, 0.0)
        for _ in range(200):
            if m.step(tgt, tgt_act, supported=True):
                break
            tgt = fly(tgt, tgt_act)
            if m.active:
                break
        self.assertTrue(m.active)
        self.assertEqual(m.time_to_active(tgt), 0.0)

    def test_a_dead_missile_stays_dead(self):
        m = self._launch()
        tgt, tgt_act = [0.0, 60_000.0, 9000.0], [250.0, 0.0, SOUTH]
        m.step(tgt, tgt_act, supported=False)
        self.assertFalse(m.alive)
        self.assertEqual(m.step(tgt, tgt_act, supported=True), "no_lock")


if __name__ == "__main__":
    unittest.main()
