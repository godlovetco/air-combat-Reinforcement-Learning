import math
import unittest

from dcs_bridge import formation as form


class StationGeometryTest(unittest.TestCase):
    def test_right_wing_is_east_of_a_north_bound_lead(self):
        # Lead at origin heading north (000): a right-side abeam station
        # (line_abreast, rel bearing 090) sits due east.
        st = form.station_position((0.0, 0.0, 3000.0), 0.0,
                                   name="line_abreast", side="right")
        rng = form.FORMATIONS["line_abreast"][1]
        self.assertAlmostEqual(st[0], rng, delta=1.0)   # east
        self.assertAlmostEqual(st[1], 0.0, delta=1.0)   # no north offset
        self.assertAlmostEqual(st[2], 3000.0, delta=1.0)

    def test_left_side_mirrors_to_the_west(self):
        st = form.station_position((0.0, 0.0, 3000.0), 0.0,
                                   name="line_abreast", side="left")
        rng = form.FORMATIONS["line_abreast"][1]
        self.assertAlmostEqual(st[0], -rng, delta=1.0)  # west

    def test_station_rotates_with_lead_heading(self):
        # Lead heading east (090): a right-side abeam station is due south.
        st = form.station_position((0.0, 0.0, 3000.0), 90.0,
                                   name="line_abreast", side="right")
        rng = form.FORMATIONS["line_abreast"][1]
        self.assertAlmostEqual(st[0], 0.0, delta=1.0)
        self.assertAlmostEqual(st[1], -rng, delta=1.0)  # south

    def test_combat_spread_stacks_high(self):
        st = form.station_position((0.0, 0.0, 3000.0), 0.0,
                                   name="combat_spread", side="right")
        self.assertGreater(st[2], 3000.0)

    def test_scout_position_is_ahead_of_lead(self):
        pt = form.scout_position((0.0, 0.0, 3000.0), 0.0)
        self.assertAlmostEqual(pt[0], 0.0, delta=1.0)
        self.assertAlmostEqual(pt[1], form.SCOUT_AHEAD_M, delta=1.0)

    def test_unknown_formation_and_side_raise(self):
        with self.assertRaises(ValueError):
            form.station_offset("finger_four")
        with self.assertRaises(ValueError):
            form.station_offset("wedge", side="up")


class StationErrorTest(unittest.TestCase):
    def test_in_position_within_tolerance(self):
        station = (1000.0, 0.0, 3000.0)
        self.assertTrue(form.in_position((1100.0, 0.0, 3000.0), station))
        self.assertFalse(form.in_position((2000.0, 0.0, 3000.0), station))


class FormationSpeedTest(unittest.TestCase):
    def test_speeds_up_when_lagging_behind(self):
        # Lead heading north; station 1 km ahead of the CCA -> catch up.
        station = (0.0, 1000.0, 3000.0)
        own = (0.0, 0.0, 3000.0)
        v = form.formation_speed(250.0, own, station, lead_heading=0.0)
        self.assertGreater(v, 250.0)

    def test_eases_off_when_ahead(self):
        station = (0.0, 0.0, 3000.0)
        own = (0.0, 1000.0, 3000.0)  # CCA is ahead of its station
        v = form.formation_speed(250.0, own, station, lead_heading=0.0)
        self.assertLess(v, 250.0)

    def test_never_commands_below_floor(self):
        station = (0.0, -100000.0, 3000.0)  # far behind: huge negative delta
        own = (0.0, 0.0, 3000.0)
        v = form.formation_speed(130.0, own, station, lead_heading=0.0)
        self.assertGreaterEqual(v, 120.0)

    def test_delta_is_bounded(self):
        station = (0.0, 100000.0, 3000.0)
        own = (0.0, 0.0, 3000.0)
        v = form.formation_speed(250.0, own, station, lead_heading=0.0)
        self.assertLessEqual(v, 250.0 + 80.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
