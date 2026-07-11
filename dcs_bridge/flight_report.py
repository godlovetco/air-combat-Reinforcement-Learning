"""Post-flight validation report from a run_pilot --log-csv engagement log.

Yoo, Kim & Shim (ICCAS 2021) validated their DCS AI pilot by analyzing
recorded engagements — target-tracking plots and closure statistics.  This
tool does the same for this addon's flight logs::

    python -m dcs_bridge.flight_report flight_01.csv
    python -m dcs_bridge.flight_report flight_01.csv --plot flight_01.png

The text summary needs only the standard library; ``--plot`` additionally
requires matplotlib (``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from typing import List, Optional

from .run_pilot import GUN_ASPECT, GUN_RANGE

CONVERSION_ASPECT = 30.0  # deg, own aspect for "pointing at the bandit"


def load_log(path: str) -> List[dict]:
    """Parse a run_pilot CSV log into typed rows (bandit fields may be None)."""
    rows: List[dict] = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {"t": float(raw["t"]), "mode": raw.get("mode", "engage"),
                   "trigger": int(raw.get("trigger") or 0)}
            for key in ("x_r", "y_r", "z_r", "x_b", "y_b", "z_b",
                        "q_r", "q_b", "range", "gamma_cmd", "psi_cmd"):
                value = raw.get(key, "")
                row[key] = float(value) if value not in ("", None) else None
            rows.append(row)
    if not rows:
        raise ValueError(f"{path!r} contains no data rows")
    return rows


def summarize(rows: List[dict]) -> dict:
    """Engagement statistics in the spirit of the paper's validation section."""
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    duration = t1 - t0

    contact = [r for r in rows if r["range"] is not None]
    ranges = [r["range"] for r in contact]
    tracking = [r for r in contact if r["q_r"] is not None and r["q_r"] < CONVERSION_ASPECT]
    in_envelope = [
        r for r in contact
        if r["range"] < GUN_RANGE and r["q_r"] is not None and r["q_r"] < GUN_ASPECT
    ]
    dt = duration / max(1, len(rows) - 1)  # nominal tick length

    first_envelope: Optional[float] = in_envelope[0]["t"] - t0 if in_envelope else None
    modes = Counter(r["mode"] for r in rows)

    return {
        "duration_s": round(duration, 1),
        "samples": len(rows),
        "contact_fraction": round(len(contact) / len(rows), 3),
        "min_range_m": round(min(ranges), 1) if ranges else None,
        "mean_range_m": round(sum(ranges) / len(ranges), 1) if ranges else None,
        "tracking_fraction": round(len(tracking) / len(contact), 3) if contact else None,
        "gun_envelope_time_s": round(len(in_envelope) * dt, 1),
        "time_to_first_gun_solution_s": (
            round(first_envelope, 1) if first_envelope is not None else None
        ),
        "trigger_time_s": round(sum(r["trigger"] for r in rows) * dt, 1),
        "modes": dict(modes),
        "own_distance_km": round(_path_length(rows, "x_r", "y_r", "z_r") / 1000.0, 1),
    }


def _path_length(rows: List[dict], kx: str, ky: str, kz: str) -> float:
    total, prev = 0.0, None
    for r in rows:
        if r[kx] is None:
            prev = None
            continue
        cur = (r[kx], r[ky], r[kz])
        if prev is not None:
            total += math.dist(prev, cur)
        prev = cur
    return total


def print_summary(stats: dict) -> None:
    print("=== engagement summary ===")
    print(f"duration            {stats['duration_s']:8.1f} s   ({stats['samples']} samples)")
    print(f"own track length    {stats['own_distance_km']:8.1f} km")
    print(f"bandit contact      {stats['contact_fraction'] * 100:7.1f} %  of flight time")
    if stats["min_range_m"] is not None:
        print(f"range min / mean    {stats['min_range_m']:8.1f} / {stats['mean_range_m']:.1f} m")
        print(f"tracking (<{CONVERSION_ASPECT:.0f} deg) {stats['tracking_fraction'] * 100:7.1f} %  of contact time")
    print(f"gun envelope time   {stats['gun_envelope_time_s']:8.1f} s"
          + (f"   (first solution at {stats['time_to_first_gun_solution_s']} s)"
             if stats["time_to_first_gun_solution_s"] is not None else ""))
    print(f"trigger held        {stats['trigger_time_s']:8.1f} s")
    print("modes               "
          + ", ".join(f"{m}: {n}" for m, n in sorted(stats["modes"].items())))


def plot(rows: List[dict], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    own = [(r["x_r"], r["y_r"], r["z_r"]) for r in rows if r["x_r"] is not None]
    bnd = [(r["x_b"], r["y_b"], r["z_b"]) for r in rows if r["x_b"] is not None]
    if own:
        ax.plot(*zip(*own), color="tab:blue", label="own (UCAV)")
    if bnd:
        ax.plot(*zip(*bnd), color="tab:red", label="bandit")
    ax.set_title("trajectories (ENU)")
    ax.legend()

    t = [r["t"] for r in rows]
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, [r["range"] if r["range"] is not None else float("nan") for r in rows])
    ax2.axhline(GUN_RANGE, color="tab:red", linestyle="--", label=f"gun range {GUN_RANGE:.0f} m")
    ax2.set_title("slant range [m]")
    ax2.set_xlabel("t [s]")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, [r["q_r"] if r["q_r"] is not None else float("nan") for r in rows],
             label="own aspect q_r")
    ax3.plot(t, [r["q_b"] if r["q_b"] is not None else float("nan") for r in rows],
             label="bandit aspect q_b")
    ax3.axhline(CONVERSION_ASPECT, color="gray", linestyle=":")
    ax3.set_title("aspect angles [deg]")
    ax3.set_xlabel("t [s]")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t, [r["z_r"] if r["z_r"] is not None else float("nan") for r in rows],
             label="own altitude")
    ax4.plot(t, [r["z_b"] if r["z_b"] is not None else float("nan") for r in rows],
             label="bandit altitude")
    trig_t = [r["t"] for r in rows if r["trigger"]]
    if trig_t:
        ax4.scatter(trig_t, [r["z_r"] for r in rows if r["trigger"]],
                    color="tab:red", s=12, zorder=5, label="trigger")
    ax4.set_title("altitude [m]")
    ax4.set_xlabel("t [s]")
    ax4.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"plot written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_csv", help="CSV written by run_pilot --log-csv")
    parser.add_argument("--plot", metavar="PNG",
                        help="also write validation plots (requires matplotlib)")
    args = parser.parse_args()

    rows = load_log(args.log_csv)
    print_summary(summarize(rows))
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
