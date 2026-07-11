"""Real-time UCAV AI pilot for DCS World.

Run this on the same machine as DCS (or point ``--dcs-host`` at it) while a
mission with a player-controlled aircraft is active and the UCAVPilot export
script is installed::

    python -m dcs_bridge.run_pilot --checkpoint checkpoints/ucav_policy.npz

Loop, at telemetry rate (~20 Hz, the cadence used for AI-pilot validation in
DCS by Yoo, Kim & Shim, ICCAS 2021):

1. read own-ship + nearest-bandit telemetry from the Lua export script,
2. fit the bandit's recent track to a constant-turn-rate model and predict
   its future positions (the paper's target-trajectory-prediction step),
3. every ``--decision-period`` seconds, feed the legacy 72-dim "what would
   each maneuver lead to" input to the Q-network and pick a maneuver
   (delta climb angle / delta heading),
4. run the bank-to-turn autopilot toward the commanded climb angle,
   heading and speed, and send stick/throttle axes back into DCS.

Without a checkpoint (or with ``--heuristic``) it flies lead pursuit onto
the predicted intercept point, which is useful for tuning autopilot gains
before trusting the network.

``--log-csv`` writes a per-tick engagement log (positions, aspect angles,
range, prediction) for post-flight validation analysis and plots.

Weapons release is OFF by default; pass ``--weapons`` to let the pilot pull
the trigger inside the gun envelope.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Optional, Tuple

from . import geometry as geo
from .autopilot import Autopilot, AutopilotConfig
from .link import DCSLink, Telemetry
from .policy import QNetwork
from .predictor import TurnRatePredictor

GUN_RANGE = 1_200.0   # m, trigger envelope
GUN_ASPECT = 4.0      # deg


def bandit_command_estimate(
    predictor: TurnRatePredictor, telem: Telemetry
) -> Tuple[float, float, float]:
    """Bandit (v, gamma, psi) estimate from its track, or attitude fallback."""
    state = predictor.motion_state()
    if state is None:
        b = telem.bandit
        return (200.0, b.pitch, b.heading)
    _, speed, heading, _, climb = state
    total = math.hypot(speed, climb)
    gamma = math.degrees(math.asin(max(-1.0, min(1.0, climb / max(total, 1.0)))))
    return (total, gamma, heading)


def steer_to_point(telem: Telemetry, point) -> Tuple[float, float]:
    """Maneuver targets (gamma_cmd, psi_cmd) that point the velocity at a point."""
    own = telem.own
    dx = point[0] - own.pos[0]
    dy = point[1] - own.pos[1]
    dz = point[2] - own.pos[2]
    horiz = math.sqrt(dx * dx + dy * dy)
    psi_cmd = geo.wrap_heading(math.degrees(math.atan2(dx, dy)))
    gamma_cmd = math.degrees(math.atan2(dz, max(horiz, 1.0)))
    return (max(-45.0, min(45.0, gamma_cmd)), psi_cmd)


def own_gamma(tas: float, vv: float) -> float:
    if tas < 1.0:
        return 0.0
    return math.degrees(math.asin(max(-1.0, min(1.0, vv / tas))))


def main() -> None:
    args = build_parser().parse_args()

    net: Optional[QNetwork] = None
    if not args.heuristic:
        if os.path.exists(args.checkpoint):
            net = QNetwork.load(args.checkpoint)
            print(f"policy loaded from {args.checkpoint}")
        else:
            print(f"checkpoint {args.checkpoint!r} not found -- flying heuristic lead pursuit")

    link = DCSLink(
        telemetry_port=args.telemetry_port,
        dcs_host=args.dcs_host,
        command_port=args.command_port,
    )
    ap = Autopilot(AutopilotConfig(invert_pitch=not args.no_invert_pitch))
    predictor = TurnRatePredictor()

    log_writer = None
    log_file = None
    if args.log_csv:
        log_file = open(args.log_csv, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["t", "x_r", "y_r", "z_r", "x_b", "y_b", "z_b",
             "q_r", "q_b", "range", "gamma_cmd", "psi_cmd", "trigger"]
        )

    gamma_cmd, psi_cmd = 0.0, 0.0
    have_cmd = False
    last_decision = -1e9
    last_log = 0.0

    print(
        f"listening for DCS telemetry on udp/{args.telemetry_port}, "
        f"sending commands to {args.dcs_host}:{args.command_port} "
        f"(weapons {'ENABLED' if args.weapons else 'disabled'})"
    )
    try:
        while True:
            telem = link.receive()
            if telem is None:
                if time.time() - last_log > 5.0:
                    print("waiting for telemetry... (is the mission running?)")
                    last_log = time.time()
                continue

            own = telem.own
            g_own = own_gamma(own.tas, own.vv)
            if not have_cmd:
                gamma_cmd, psi_cmd = g_own, own.heading
                have_cmd = True

            trigger = 0
            feats = None
            if telem.bandit is not None:
                predictor.update(telem.t, telem.bandit.pos)
                act_b = bandit_command_estimate(predictor, telem)
                feats = geo.situation(
                    list(own.pos), [own.tas, g_own, own.heading],
                    list(telem.bandit.pos), list(act_b),
                )

            if telem.bandit is None:
                # No hostile airborne: hold altitude in a gentle right orbit.
                gamma_cmd = 0.0
                psi_cmd = geo.wrap_heading(own.heading + 15.0)
            elif telem.t - last_decision >= args.decision_period:
                last_decision = telem.t
                lead = predictor.intercept_point(own.pos, own.tas)
                if net is not None:
                    pred_next = predictor.predict(args.decision_period)
                    x = geo.build_network_input(
                        list(own.pos), [own.tas, gamma_cmd, psi_cmd],
                        list(telem.bandit.pos), list(act_b),
                        dt=args.decision_period,
                        next_pos_b=pred_next,
                    )
                    choice = net.act(x)
                    d_gamma, d_psi = geo.ACTION_DELTAS[choice]
                    gamma_cmd = max(-geo.GAMMA_LIMIT_DEG,
                                    min(geo.GAMMA_LIMIT_DEG, gamma_cmd + d_gamma))
                    psi_cmd = geo.wrap_heading(psi_cmd + d_psi)
                else:
                    aim = lead if lead is not None else telem.bandit.pos
                    gamma_cmd, psi_cmd = steer_to_point(telem, aim)

            if args.weapons and feats is not None:
                if feats[2] < GUN_RANGE and feats[0] < GUN_ASPECT:
                    trigger = 1

            controls = ap.command(
                t=telem.t,
                pitch_deg=own.pitch,
                bank_deg=own.bank,
                heading_deg=own.heading,
                tas=own.tas,
                gamma_cmd_deg=gamma_cmd,
                psi_cmd_deg=psi_cmd,
                v_cmd=args.target_speed,
            )
            controls.trigger = trigger
            link.send(controls)

            if log_writer is not None:
                b = telem.bandit
                log_writer.writerow([
                    round(telem.t, 3), *[round(v, 1) for v in own.pos],
                    *([round(v, 1) for v in b.pos] if b else ["", "", ""]),
                    round(feats[0], 2) if feats else "",
                    round(feats[1], 2) if feats else "",
                    round(feats[2], 1) if feats else "",
                    round(gamma_cmd, 1), round(psi_cmd, 1), trigger,
                ])

            if telem.t - last_log >= 1.0:
                last_log = telem.t
                bandit_txt = "none"
                if telem.bandit is not None:
                    d = math.dist(own.pos, telem.bandit.pos)
                    bandit_txt = f"{telem.bandit.name} @ {d/1000.0:.1f} km"
                print(
                    f"t={telem.t:8.1f}  tas={own.tas:5.1f}  alt={own.pos[2]:6.0f}  "
                    f"hdg={own.heading:5.1f}->{psi_cmd:5.1f}  "
                    f"gamma={g_own:5.1f}->{gamma_cmd:5.1f}  bandit: {bandit_txt}"
                    + ("  FIRING" if trigger else "")
                )
    except KeyboardInterrupt:
        print("\nAI pilot stopped")
    finally:
        if log_file is not None:
            log_file.close()
        link.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/ucav_policy.npz")
    p.add_argument("--heuristic", action="store_true",
                   help="ignore the checkpoint and fly lead pursuit")
    p.add_argument("--decision-period", type=float, default=0.5,
                   help="seconds between policy decisions")
    p.add_argument("--target-speed", type=float, default=250.0,
                   help="TAS the throttle loop holds, m/s")
    p.add_argument("--weapons", action="store_true",
                   help="allow trigger pulls inside the gun envelope")
    p.add_argument("--log-csv", default="",
                   help="write a per-tick engagement log to this CSV for validation")
    p.add_argument("--telemetry-port", type=int, default=7778)
    p.add_argument("--command-port", type=int, default=7779)
    p.add_argument("--dcs-host", default="127.0.0.1")
    p.add_argument("--no-invert-pitch", action="store_true",
                   help="do not invert the pitch axis sign sent to DCS")
    return p


if __name__ == "__main__":
    main()
