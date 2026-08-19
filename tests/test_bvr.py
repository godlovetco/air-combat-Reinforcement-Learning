import math
import unittest

from dcs_bridge import bvr
from dcs_bridge import geometry as geo


NORTH = 0.0
SOUTH = 180.0
EAST = 90.0


def fly(pos, act, dt=1.0):
    return geo.step_point_mass(pos, act, dt)


def beam(threat_pos, own_pos, own_psi):
    """Heading that puts the threat on our 3/9 line, tracked continuously.

    A notch is held against a *moving* line of sight: as the missile flies past
    the bearing rotates, so a pilot who picks one heading and holds it slides
    out of the Doppler gate on their own.
    """
    bearing = geo.wrap_heading(math.degrees(math.atan2(
        threat_pos[0] - own_pos[0], threat_pos[1] - own_pos[1])))
    left, right = geo.wrap_heading(bearing - 90.0), geo.wrap_heading(bearing + 90.0)
    return (left if abs(geo.heading_error(left, own_psi))
            <= abs(geo.heading_error(right, own_psi)) else right)


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

    def test_holding_the_beam_on_the_missile_defeats_it(self):
        m = self._launch()
        tgt, psi = [0.0, 30_000.0, 9000.0], SOUTH
        outcome = None
        for _ in range(200):
            # Run at the missile until it goes active, then beam it -- tracking
            # the bearing every step, which is what holding a notch means.
            psi = beam(m.pos, tgt, psi) if m.active else SOUTH
            tgt_act = [250.0, 0.0, psi]
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "notched")

    def test_a_beam_heading_picked_once_and_held_does_not(self):
        """The instructive failure: the line of sight rotates, the pilot doesn't.

        Same defensive turn, except the heading is frozen at the moment the
        missile goes active. The seeker coasts, the bearing drifts, the target
        walks back out of the Doppler gate, and the round reacquires.
        """
        m = self._launch()
        tgt, frozen = [0.0, 30_000.0, 9000.0], None
        outcome = None
        for _ in range(200):
            if m.active and frozen is None:
                frozen = beam(m.pos, tgt, SOUTH)
            tgt_act = [250.0, 0.0, frozen if frozen is not None else SOUTH]
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "hit")

    def test_a_notch_broken_early_lets_the_seeker_reacquire(self):
        m = self._launch()
        tgt, psi = [0.0, 30_000.0, 9000.0], SOUTH
        held = 0
        outcome = None
        for _ in range(200):
            if m.active:
                if held < 3:
                    psi = beam(m.pos, tgt, psi)   # beam it...
                    held += 1
                else:
                    psi = geo.wrap_heading(math.degrees(math.atan2(
                        m.pos[0] - tgt[0], m.pos[1] - tgt[1])))  # ...then turn back
            tgt_act = [250.0, 0.0, psi]
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "hit")

    def test_a_coasting_seeker_reports_itself(self):
        m = self._launch()
        tgt, psi = [0.0, 20_000.0, 9000.0], SOUTH
        saw_coasting = False
        for _ in range(200):
            psi = beam(m.pos, tgt, psi) if m.active else SOUTH
            tgt_act = [250.0, 0.0, psi]
            if m.step(tgt, tgt_act, supported=True):
                break
            saw_coasting = saw_coasting or m.coasting
            tgt = fly(tgt, tgt_act)
        self.assertTrue(saw_coasting)

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


class AesaTest(unittest.TestCase):
    def setUp(self):
        self.msa = bvr.Radar.mechanical()
        self.aesa = bvr.Radar.aesa()
        self.own_pos = [0.0, 0.0, 9000.0]
        self.own_act = [250.0, 0.0, NORTH]

    def test_the_two_arrays_differ_where_it_matters(self):
        self.assertEqual(self.msa.kind, "mechanical")
        self.assertEqual(self.aesa.kind, "aesa")
        self.assertGreater(self.aesa.max_range, self.msa.max_range)
        self.assertLess(self.aesa.notch_closure, self.msa.notch_closure)
        self.assertGreater(self.aesa.simultaneous_tracks, self.msa.simultaneous_tracks)
        self.assertLess(self.aesa.reacquire_time, self.msa.reacquire_time)

    def test_aesa_sees_a_target_the_mechanical_set_cannot_reach(self):
        far = [0.0, 110_000.0, 9000.0]
        self.assertFalse(self.msa.can_see(self.own_pos, self.own_act, far,
                                          [250.0, 0.0, SOUTH]))
        self.assertTrue(self.aesa.can_see(self.own_pos, self.own_act, far,
                                          [250.0, 0.0, SOUTH]))

    def test_the_narrower_notch_holds_a_target_the_mechanical_set_drops(self):
        # A target beaming imperfectly: radial velocity lands between the two
        # Doppler gates, so the AESA keeps it and the mechanical set does not.
        tgt = [0.0, 40_000.0, 9000.0]
        offset = math.degrees(math.asin(40.0 / 250.0))   # ~40 m/s of radial
        heading = geo.wrap_heading(EAST + offset)
        radial = abs(bvr.radial_speed(self.own_pos, tgt, [250.0, 0.0, heading]))
        self.assertTrue(bvr.AESA_NOTCH_CLOSURE < radial < bvr.NOTCH_CLOSURE)
        self.assertFalse(self.msa.can_see(self.own_pos, self.own_act, tgt,
                                          [250.0, 0.0, heading]))
        self.assertTrue(self.aesa.can_see(self.own_pos, self.own_act, tgt,
                                          [250.0, 0.0, heading]))

    def test_a_precise_notch_still_defeats_an_aesa(self):
        tgt, tgt_act = [0.0, 40_000.0, 9000.0], [250.0, 0.0, EAST]
        self.assertTrue(self.aesa.in_notch(self.own_pos, self.own_act, tgt, tgt_act))

    def test_a_conventional_radar_is_heard_before_it_can_see(self):
        """The normal state of affairs: spiked before you are seen.

        A warning receiver only listens one way; the radar needs the return
        trip, so the emitter announces itself well outside its own detection
        range.
        """
        self.assertGreater(self.msa.warns_at(), self.msa.max_range)
        beyond = [0.0, self.msa.max_range + 20_000.0, 9000.0]
        self.assertTrue(self.msa.is_detected_by_rwr(self.own_pos, beyond))
        self.assertFalse(self.msa.can_see(self.own_pos, self.own_act, beyond,
                                          [250.0, 0.0, SOUTH]))

    def test_an_lpi_array_inverts_that(self):
        self.assertLess(self.aesa.warns_at(), self.aesa.max_range)
        # Inside the AESA's reach but outside the range it can be heard at:
        # it is tracking a target that does not know it.
        quiet = [0.0, 100_000.0, 9000.0]
        self.assertTrue(self.aesa.can_see(self.own_pos, self.own_act, quiet,
                                          [250.0, 0.0, SOUTH]))
        self.assertFalse(self.aesa.is_detected_by_rwr(self.own_pos, quiet))
        # The mechanical set at the same range is both blind and loud.
        self.assertFalse(self.msa.can_see(self.own_pos, self.own_act, quiet,
                                          [250.0, 0.0, SOUTH]))
        self.assertTrue(self.msa.is_detected_by_rwr(self.own_pos, quiet))

    def test_the_aesa_is_eventually_detected_too(self):
        close = [0.0, 20_000.0, 9000.0]
        self.assertTrue(self.aesa.is_detected_by_rwr(self.own_pos, close))

    def test_presets_accept_overrides(self):
        self.assertEqual(bvr.Radar.aesa(max_range=200_000.0).max_range, 200_000.0)
        self.assertEqual(bvr.Radar.aesa(max_range=200_000.0).kind, "aesa")


class MissileCatalogTest(unittest.TestCase):
    def test_catalog_contents(self):
        self.assertEqual(set(bvr.MISSILES), {"ARH-medium", "ARH-long", "IR-short"})
        self.assertIs(bvr.DEFAULT_MISSILE, bvr.ARH_MEDIUM)

    def test_the_ramjet_has_a_much_larger_no_escape_zone(self):
        self.assertGreater(bvr.ARH_LONG.rne, bvr.ARH_MEDIUM.rne * 2)
        self.assertGreater(bvr.ARH_LONG.max_flight_time,
                           bvr.ARH_MEDIUM.max_flight_time)

    def test_infrared_is_fire_and_forget(self):
        self.assertTrue(bvr.IR_SHORT.fire_and_forget)
        self.assertFalse(bvr.ARH_MEDIUM.fire_and_forget)
        m = bvr.Missile.launch(bvr.IR_SHORT, "agent", [0.0, 0.0, 9000.0],
                               [250.0, 0.0, NORTH])
        self.assertTrue(m.active)   # guiding off the rail, no support needed

    def test_infrared_cannot_be_notched(self):
        m = bvr.Missile.launch(bvr.IR_SHORT, "agent", [0.0, 0.0, 9000.0],
                               [250.0, 0.0, NORTH])
        tgt, psi = [0.0, 12_000.0, 9000.0], EAST
        outcome = None
        for _ in range(120):
            psi = beam(m.pos, tgt, psi)      # a perfect notch, held all the way
            tgt_act = [250.0, 0.0, psi]
            outcome = m.step(tgt, tgt_act, supported=False)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "hit")

    def test_an_active_radar_missile_would_have_been_notched_there(self):
        m = bvr.Missile.launch(bvr.ARH_MEDIUM, "agent", [0.0, 0.0, 9000.0],
                               [250.0, 0.0, NORTH])
        tgt, psi = [0.0, 12_000.0, 9000.0], EAST
        outcome = None
        for _ in range(120):
            psi = beam(m.pos, tgt, psi)
            tgt_act = [250.0, 0.0, psi]
            outcome = m.step(tgt, tgt_act, supported=True)
            if outcome:
                break
            tgt = fly(tgt, tgt_act)
        self.assertEqual(outcome, "notched")

    def test_seeker_is_built_from_the_spec(self):
        seeker = bvr.ARH_LONG.seeker()
        self.assertEqual(seeker.max_range, bvr.ARH_LONG.seeker_range)
        self.assertEqual(seeker.gimbal_deg, bvr.ARH_LONG.seeker_gimbal_deg)


class WezTest(unittest.TestCase):
    def _wez(self, spec, tgt_heading, rng=40_000.0):
        return bvr.weapon_engagement_zone(
            spec, [0.0, 0.0, 9000.0], [250.0, 0.0, NORTH],
            [0.0, rng, 9000.0], [250.0, 0.0, tgt_heading])

    def test_bands_are_ordered(self):
        w = self._wez(bvr.ARH_MEDIUM, SOUTH)
        self.assertGreater(w.rmax, w.rne)
        self.assertGreater(w.rne, w.rmin)
        self.assertLessEqual(w.rtr, w.rmax)

    def test_rtr_is_the_reach_against_a_target_that_reverses(self):
        head_on = self._wez(bvr.ARH_MEDIUM, SOUTH)
        running = self._wez(bvr.ARH_MEDIUM, NORTH)
        self.assertAlmostEqual(head_on.rtr, running.rmax, places=6)

    def test_a_shot_the_target_can_run_from(self):
        w = self._wez(bvr.ARH_MEDIUM, SOUTH, rng=40_000.0)
        self.assertTrue(w.in_envelope)
        self.assertFalse(w.in_nez)
        self.assertTrue(w.defeatable_by_running)   # 40 km > Rtr

    def test_a_shot_that_cannot_be_run_from(self):
        w = self._wez(bvr.ARH_MEDIUM, SOUTH, rng=15_000.0)
        self.assertTrue(w.in_nez)
        self.assertFalse(w.defeatable_by_running)

    def test_the_ramjet_makes_running_stop_working(self):
        medium = self._wez(bvr.ARH_MEDIUM, SOUTH, rng=40_000.0)
        ramjet = self._wez(bvr.ARH_LONG, SOUTH, rng=40_000.0)
        self.assertTrue(medium.defeatable_by_running)
        self.assertFalse(ramjet.defeatable_by_running)
        self.assertTrue(ramjet.in_nez)

    def test_time_of_flight_shrinks_as_the_range_closes(self):
        near = self._wez(bvr.ARH_MEDIUM, SOUTH, rng=10_000.0)
        far = self._wez(bvr.ARH_MEDIUM, SOUTH, rng=60_000.0)
        self.assertLess(near.time_of_flight, far.time_of_flight)

    def test_launch_authority_is_the_same_thing_as_a_dict(self):
        d = bvr.launch_authority(bvr.ARH_MEDIUM, [0.0, 0.0, 9000.0],
                                 [250.0, 0.0, NORTH], [0.0, 20_000.0, 9000.0],
                                 [250.0, 0.0, SOUTH])
        for key in ("range", "rmax", "rtr", "rne", "rmin", "in_envelope", "in_nez"):
            self.assertIn(key, d)
        self.assertTrue(d["in_nez"])
