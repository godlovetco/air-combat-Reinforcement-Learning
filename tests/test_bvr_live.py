import math
import unittest

from dcs_bridge import bvr
from dcs_bridge import geometry as geo
from dcs_bridge.bvr_live import BVRController, Shot, own_gamma_from
from dcs_bridge.link import Contact, Ownship, RwrContact, Sensors, Telemetry


def telem(t=0.0, own_pos=(0.0, 0.0, 8000.0), heading=0.0, tas=250.0, vv=0.0,
          bandit_pos=(0.0, 40_000.0, 8000.0), bandit_heading=180.0,
          sensors=None):
    own = Ownship("me", own_pos, heading, 0.0, 0.0, tas, vv)
    bandit = (None if bandit_pos is None else
              Contact("MiG-29S", bandit_pos, bandit_heading, 0.0))
    return Telemetry(t=t, own=own, bandit=bandit, sensors=sensors)


def sensors(locked=True, missiles=4, threats=()):
    return Sensors(locked=locked, missiles=missiles, threats=tuple(threats))


class ShotTest(unittest.TestCase):
    def setUp(self):
        self.shot = Shot(bvr.ARH_MEDIUM, launched_at=0.0, launch_range=40_000.0)

    def test_support_owed_falls_as_the_round_flies(self):
        first = self.shot.support_owed(1.0, 40_000.0)
        later = self.shot.support_owed(10.0, 40_000.0)
        self.assertGreater(first, later)
        self.assertGreater(first, 0.0)

    def test_it_goes_active_once_it_is_inside_the_activation_range(self):
        self.assertFalse(self.shot.active(1.0, 40_000.0))
        # 30 s at 900 m/s is 27 km of travel: well inside 16 km to run.
        self.assertTrue(self.shot.active(30.0, 40_000.0))

    def test_a_closing_target_shortens_the_support(self):
        far = self.shot.support_owed(5.0, 40_000.0)
        closing = self.shot.support_owed(5.0, 33_000.0)
        self.assertLess(closing, far)

    def test_it_expires_when_the_energy_runs_out(self):
        self.assertFalse(self.shot.expired(60.0))
        self.assertTrue(self.shot.expired(bvr.ARH_MEDIUM.max_flight_time + 1.0))

    def test_range_never_goes_negative(self):
        self.assertGreaterEqual(self.shot.range_to_target(1000.0, 100.0), 0.0)


class GammaTest(unittest.TestCase):
    def test_climb_and_dive(self):
        self.assertAlmostEqual(own_gamma_from(Ownship("", (0, 0, 0), 0, 0, 0,
                                                      250.0, 125.0)), 30.0,
                               places=3)
        self.assertLess(own_gamma_from(Ownship("", (0, 0, 0), 0, 0, 0,
                                               250.0, -125.0)), 0.0)

    def test_a_stopped_aircraft_does_not_divide_by_zero(self):
        self.assertEqual(own_gamma_from(Ownship("", (0, 0, 0), 0, 0, 0, 0.0,
                                                10.0)), 0.0)


class SituationTest(unittest.TestCase):
    def setUp(self):
        self.c = BVRController()

    def test_block_shape(self):
        sit = self.c.situation(telem(sensors=sensors()))
        block = sit["bvr"]
        self.assertTrue(block["locked"])
        self.assertEqual(block["missiles"], 4)
        self.assertFalse(block["shot_in_flight"])
        self.assertIn("in_nez", block["wez"])
        self.assertAlmostEqual(sit["target_bearing_deg"], 0.0)

    def test_no_sensor_block_means_no_bvr_picture(self):
        block = self.c.situation(telem(sensors=None))["bvr"]
        self.assertFalse(block["locked"])
        self.assertFalse(block["spiked"])
        self.assertIsNone(block["threat"])
        self.assertEqual(block["missiles"], 0)

    def test_a_launch_warning_becomes_an_active_threat(self):
        rwr = [RwrContact(az=90.0, power=0.9, launch=True, lock=True)]
        block = self.c.situation(telem(sensors=sensors(threats=rwr)))["bvr"]
        self.assertIsNotNone(block["threat"])
        self.assertTrue(block["threat"]["active"])
        self.assertAlmostEqual(block["threat"]["off_nose_deg"], 90.0, places=3)

    def test_a_search_only_emitter_is_not_a_threat(self):
        rwr = [RwrContact(az=45.0, power=0.4, launch=False, lock=False)]
        self.assertIsNone(
            self.c.situation(telem(sensors=sensors(threats=rwr)))["bvr"]["threat"])

    def test_a_lock_without_a_launch_is_reported_but_not_active(self):
        rwr = [RwrContact(az=45.0, power=0.8, launch=False, lock=True)]
        threat = self.c.situation(telem(sensors=sensors(threats=rwr)))["bvr"]["threat"]
        self.assertIsNotNone(threat)
        self.assertFalse(threat["active"])

    def test_time_to_impact_is_unknown_rather_than_invented(self):
        rwr = [RwrContact(az=30.0, power=0.9, launch=True, lock=True)]
        threat = self.c.situation(telem(sensors=sensors(threats=rwr)))["bvr"]["threat"]
        self.assertIsNone(threat["seconds"])
        self.assertTrue(threat["from_rwr"])

    def test_no_bandit_still_produces_a_usable_block(self):
        sit = self.c.situation(telem(bandit_pos=None, sensors=sensors()))
        self.assertEqual(sit["bvr"]["wez"], {})
        self.assertAlmostEqual(sit["target_bearing_deg"], sit["own"]["heading_deg"])


class ShotDecisionTest(unittest.TestCase):
    def setUp(self):
        self.c = BVRController()

    def test_weapons_hold_never_shoots(self):
        d = self.c.decide(telem(sensors=sensors()), weapons_free=False)
        self.assertEqual(d.weapon, 0)
        self.assertEqual(d.reason, "weapons hold")

    def test_it_shoots_inside_the_envelope(self):
        d = self.c.decide(telem(sensors=sensors()), weapons_free=True)
        self.assertEqual(d.weapon, 1)

    def test_no_lock_no_shot(self):
        d = self.c.decide(telem(sensors=sensors(locked=False)), weapons_free=True)
        self.assertEqual(d.weapon, 0)
        self.assertEqual(d.reason, "no lock")

    def test_winchester(self):
        d = self.c.decide(telem(sensors=sensors(missiles=0)), weapons_free=True)
        self.assertEqual(d.weapon, 0)
        self.assertEqual(d.reason, "winchester")

    def test_out_of_the_envelope(self):
        far = telem(bandit_pos=(0.0, 200_000.0, 8000.0), sensors=sensors())
        d = self.c.decide(far, weapons_free=True)
        self.assertEqual(d.weapon, 0)
        self.assertEqual(d.reason, "out of the envelope")

    def test_trigger_discipline_after_a_confirmed_shot(self):
        t0 = telem(t=0.0, sensors=sensors())
        self.assertEqual(self.c.decide(t0, weapons_free=True).weapon, 1)
        self.c.confirm_shot(t0)
        d = self.c.decide(telem(t=2.0, sensors=sensors()), weapons_free=True)
        self.assertEqual(d.weapon, 0)
        self.assertEqual(d.reason, "trigger discipline")

    # The track-capacity rule only binds in a narrow window: the first shot has
    # to still need support (launched outside the seeker's activation range)
    # while the geometry is inside the no-escape zone, so the shot-quality rule
    # is not the thing refusing.  Trigger discipline is relaxed in these two so
    # the window is reachable at all -- it is a policy setting, not physics.
    _WINDOW = dict(bandit_pos=(0.0, 23_000.0, 8000.0))

    def test_a_mechanical_radar_can_only_guide_one_round(self):
        c = BVRController(min_shot_interval=2.0)
        c.confirm_shot(telem(t=0.0, **self._WINDOW, sensors=sensors()))
        self.assertFalse(c.shots[0].active(2.0, 23_000.0))   # still supported
        d = c.decide(telem(t=2.0, **self._WINDOW, sensors=sensors()),
                     weapons_free=True)
        self.assertEqual(d.weapon, 0)
        self.assertIn("supporting", d.reason)

    def test_an_aesa_can_guide_a_second_round_there(self):
        c = BVRController(radar=bvr.Radar.aesa(), min_shot_interval=2.0)
        c.confirm_shot(telem(t=0.0, **self._WINDOW, sensors=sensors()))
        d = c.decide(telem(t=2.0, **self._WINDOW, sensors=sensors()),
                     weapons_free=True)
        self.assertEqual(d.weapon, 1)

    def test_a_shot_taken_inside_the_activation_range_needs_no_support(self):
        """Launched at 15 km with a 16 km seeker: pitbull off the rail."""
        near = dict(bandit_pos=(0.0, 15_000.0, 8000.0))
        c = BVRController(min_shot_interval=2.0)
        c.confirm_shot(telem(t=0.0, **near, sensors=sensors()))
        self.assertTrue(c.shots[0].active(0.0, 15_000.0))
        d = c.decide(telem(t=2.0, **near, sensors=sensors()), weapons_free=True)
        self.assertEqual(d.weapon, 1)   # the radar was never tied up

    def test_multi_track_does_not_license_stacking_defeatable_shots(self):
        """An AESA guides more rounds; it does not make a bad shot good."""
        c = BVRController(radar=bvr.Radar.aesa())
        far = dict(bandit_pos=(0.0, 50_000.0, 8000.0))   # in range, outside NEZ
        c.confirm_shot(telem(t=0.0, **far, sensors=sensors()))
        d = c.decide(telem(t=20.0, **far, sensors=sensors()), weapons_free=True)
        self.assertEqual(d.weapon, 0)
        self.assertIn("defeatable", d.reason)

    def test_a_shot_that_has_gone_active_frees_the_radar(self):
        t0 = telem(t=0.0, sensors=sensors())
        self.c.confirm_shot(t0)
        # 40 s at 900 m/s: long past pitbull, so the launcher is free again.
        d = self.c.decide(telem(t=40.0, sensors=sensors()), weapons_free=True)
        self.assertEqual(d.weapon, 1)

    def test_a_defeatable_shot_is_taken_when_nothing_is_in_the_air(self):
        far = telem(bandit_pos=(0.0, 50_000.0, 8000.0), sensors=sensors())
        d = self.c.decide(far, weapons_free=True)
        self.assertEqual(d.weapon, 1)
        self.assertEqual(d.reason, "in the envelope")

    def test_the_maneuver_defends_before_it_shoots(self):
        rwr = [RwrContact(az=90.0, power=0.9, launch=True, lock=True)]
        near = telem(bandit_pos=(0.0, 15_000.0, 8000.0),
                     sensors=sensors(threats=rwr))
        d = self.c.decide(near, weapons_free=True)
        # Threat on the right: the timeline turns to put it on the beam.
        self.assertAlmostEqual(abs(geo.heading_error(d.desired_heading, 90.0)),
                               90.0, places=3)


if __name__ == "__main__":
    unittest.main()
