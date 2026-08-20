import os
import tempfile
import unittest

from dcs_bridge.flight_report import load_log, summarize

HEADER = "t,mode,x_r,y_r,z_r,x_b,y_b,z_b,q_r,q_b,range,gamma_cmd,psi_cmd,trigger"


def make_log(rows):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as fh:
        fh.write(HEADER + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return path


class FlightReportTest(unittest.TestCase):
    def test_summary_statistics(self):
        # 4 ticks, 1 s apart: closing from 5 km to inside the gun envelope.
        rows = [
            # t, mode, own xyz, bandit xyz, q_r, q_b, range, gamma, psi, trig
            [0.0, "engage", 0, 0, 3000, 0, 5000, 3000, 45.0, 170.0, 5000.0, 0, 0, 0],
            [1.0, "engage", 0, 250, 3000, 0, 4800, 3000, 20.0, 175.0, 4550.0, 0, 0, 0],
            [2.0, "engage", 0, 500, 3000, 0, 4600, 3000, 5.0, 178.0, 4100.0, 0, 0, 0],
            [3.0, "engage", 0, 750, 3000, 0, 1800, 3000, 2.0, 179.0, 1050.0, 0, 0, 1],
        ]
        path = make_log(rows)
        try:
            stats = summarize(load_log(path))
        finally:
            os.unlink(path)

        self.assertEqual(stats["duration_s"], 3.0)
        self.assertEqual(stats["samples"], 4)
        self.assertEqual(stats["contact_fraction"], 1.0)
        self.assertEqual(stats["min_range_m"], 1050.0)
        # aspect < 30 deg in 3 of 4 contact samples
        self.assertEqual(stats["tracking_fraction"], 0.75)
        # one sample inside range<1200 and q_r<4 -> one tick of envelope time
        self.assertEqual(stats["gun_envelope_time_s"], 1.0)
        self.assertEqual(stats["time_to_first_gun_solution_s"], 3.0)
        self.assertEqual(stats["trigger_time_s"], 1.0)
        self.assertEqual(stats["modes"], {"engage": 4})
        # own ship moved 750 m north
        self.assertEqual(stats["own_distance_km"], 0.8)

    def test_handles_missing_bandit(self):
        rows = [
            [0.0, "anchor", 0, 0, 3000, "", "", "", "", "", "", 0, 90, 0],
            [1.0, "anchor", 0, 100, 3000, "", "", "", "", "", "", 0, 110, 0],
        ]
        path = make_log(rows)
        try:
            stats = summarize(load_log(path))
        finally:
            os.unlink(path)
        self.assertEqual(stats["contact_fraction"], 0.0)
        self.assertIsNone(stats["min_range_m"])
        self.assertEqual(stats["gun_envelope_time_s"], 0.0)
        self.assertEqual(stats["modes"], {"anchor": 2})

    def test_empty_log_raises(self):
        path = make_log([])
        try:
            with self.assertRaises(ValueError):
                load_log(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
