"""Real-time UCAV AI pilot for DCS World.

Run this on the same machine as DCS (or point ``--dcs-host`` at it) while a
mission with a player-controlled aircraft is active and the UCAVPilot export
script is installed::

    python -m dcs_bridge.run_pilot --checkpoint checkpoints/ucav_policy.npz --radio

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

``--wso`` flips the roles: the *human* flies and the AI rides in back as
the Weapon Systems Officer, calling the fight as text — threat warnings, BRA
calls, weapons cues, energy/geometry tips, and the trained policy's
recommended maneuver in plain words ("recommend come right to 210, nose
up"). In WSO mode the AI never sends stick/throttle; the pilot keeps the
jet.

``--radio`` starts the LLM radio wingman: the flight lead speaks (or types)
commands in Korean or English; a Claude model turns them into tactical
orders (engage / anchor / vector / break / RTB / weapons state) and answers
with a short brevity call over TTS.  The AI also makes proactive calls
(tally, in guns, blind).  Without an Anthropic API key the radio falls back
to an offline brevity-code parser.

Without a checkpoint (or with ``--heuristic``) it flies lead pursuit onto
the predicted intercept point, which is useful for tuning autopilot gains
before trusting the network.

``--log-csv`` writes a per-tick engagement log (positions, aspect angles,
range, prediction) for post-flight validation analysis and plots.

Weapons release is OFF by default; pass ``--weapons`` (or radio
"weapons free") to let the pilot pull the trigger inside the gun envelope.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import threading
import time
from typing import Optional, Tuple

from . import config as cfg
from . import formation as form
from . import geometry as geo
from . import licensing
from .logging_setup import DEFAULT_LOG_FILE, setup_logging
from .autopilot import Autopilot, AutopilotConfig
from .link import Contact, DCSLink, Telemetry
from .orders import PilotState
from .policy import QNetwork
from .predictor import TurnRatePredictor
from .wso import WSOAdvisor, make_wso

GUN_RANGE = 1_200.0   # m, trigger envelope
GUN_ASPECT = 4.0      # deg


def lead_speed_estimate(
    predictor: TurnRatePredictor, lead: Contact, fallback: float
) -> float:
    """Best available TAS for the crewed lead: datalink value, else track fit."""
    if lead.tas is not None:
        return lead.tas
    state = predictor.motion_state()
    if state is not None:
        _, speed, _, _, climb = state
        return math.hypot(speed, climb)
    return fallback


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


def steer_to_point(own_pos, point) -> Tuple[float, float]:
    """Maneuver targets (gamma_cmd, psi_cmd) that point the velocity at a point."""
    dx = point[0] - own_pos[0]
    dy = point[1] - own_pos[1]
    dz = point[2] - own_pos[2]
    horiz = math.sqrt(dx * dx + dy * dy)
    psi_cmd = geo.wrap_heading(math.degrees(math.atan2(dx, dy)))
    gamma_cmd = math.degrees(math.atan2(dz, max(horiz, 1.0)))
    return (max(-45.0, min(45.0, gamma_cmd)), psi_cmd)


def own_gamma(tas: float, vv: float) -> float:
    if tas < 1.0:
        return 0.0
    return math.degrees(math.asin(max(-1.0, min(1.0, vv / tas))))


def altitude_hold_gamma(target_alt: Optional[float], current_alt: float) -> float:
    if target_alt is None:
        return 0.0
    return max(-20.0, min(20.0, 0.02 * (target_alt - current_alt)))


def build_wso_situation(telem, own, g_own, feats, gamma_cmd, psi_cmd, mode, snap) -> dict:
    """Assemble the tactical picture the WSO back-seater reasons over."""
    bandit = None
    recommend = None
    if telem.bandit is not None and feats is not None:
        bearing = geo.wrap_heading(math.degrees(math.atan2(
            telem.bandit.pos[0] - own.pos[0],
            telem.bandit.pos[1] - own.pos[1],
        )))
        bandit = {
            "name": telem.bandit.name,
            "range_m": feats[2],
            "own_aspect_deg": feats[0],
            "bandit_aspect_deg": feats[1],
            "bearing_deg": bearing,
            "altitude_m": telem.bandit.pos[2],
        }
        if mode == "engage":
            recommend = {"heading_deg": psi_cmd, "gamma_deg": gamma_cmd}

    lead = None
    if telem.lead is not None:
        station_error = 0.0
        if mode == "formation":
            station = form.station_position(
                telem.lead.pos, telem.lead.heading,
                snap["formation_station"], snap["formation_side"])
            station_error = form.station_error(own.pos, station)
        lead = {
            "name": telem.lead.name,
            "range_m": math.dist(own.pos, telem.lead.pos),
            "station_error_m": station_error,
        }

    return {
        "t": telem.t,
        "own": {
            "altitude_m": own.pos[2], "tas": own.tas, "vv": own.vv,
            "bank_deg": own.bank, "gamma_deg": g_own, "heading_deg": own.heading,
        },
        "bandit": bandit,
        "lead": lead,
        "weapons_free": snap["weapons_free"],
        "recommend": recommend,
    }


def start_radio(args, state: PilotState, get_situation) -> Optional["object"]:
    """Spin up the radio wingman thread; returns the VoiceIO for proactive calls."""
    from .voice import VoiceIO
    from .wingman import make_agent

    voice = VoiceIO(prefer_voice=not args.radio_text_only, language=args.radio_lang)
    agent = make_agent(model=args.radio_model, offline=args.radio_offline)

    def radio_loop() -> None:
        voice.say("Viper 2, checking in as fragged.")
        while True:
            transmission = voice.listen()
            if not transmission:
                continue
            if transmission.lower() in ("quit", "exit"):
                break
            try:
                reply, orders = agent.radio(transmission, get_situation())
            except Exception as exc:
                voice.say("2, radio garbled, say again.")
                print(f"radio: agent error: {exc}")
                continue
            for order in orders:
                state.apply(order)
                print(f"[RADIO] order applied: {order.describe()}")
            if reply:
                voice.say(reply)

    threading.Thread(target=radio_loop, daemon=True, name="radio-wingman").start()
    return voice


def parse_args(argv=None):
    """Parse CLI args, applying a ``--config`` TOML file underneath them.

    A config file supplies defaults; explicit command-line flags still win.
    """
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        valid = [a.dest for a in parser._actions if a.dest not in ("help", "version")]
        try:
            overrides = cfg.load_config(known.config, valid_keys=valid)
        except cfg.ConfigError as exc:
            parser.error(str(exc))
        parser.set_defaults(**overrides)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    log = setup_logging(args.log_level, args.log_file or None)
    if getattr(args, "config", None):
        log.info("loaded config from %s", args.config)

    # ---- product license: the LLM tier is a paid feature ---------------
    lic = licensing.load_license(args.license_key)
    log.info("license: %s", lic.describe())
    if not lic.has(licensing.FEATURE_LLM):
        if args.radio and not args.radio_offline:
            log.warning("radio: the LLM wingman is a Pro feature; using the "
                        "offline brevity parser (enter a license key to unlock).")
            args.radio_offline = True
        if args.wso and args.wso_llm:
            log.warning("WSO: the LLM back-seater is a Pro feature; using the "
                        "rule-based advisor (enter a license key to unlock).")
            args.wso_llm = False

    net: Optional[QNetwork] = None
    if not args.heuristic:
        if os.path.exists(args.checkpoint):
            net = QNetwork.load(args.checkpoint)
            log.info("policy loaded from %s", args.checkpoint)
        else:
            log.warning("checkpoint %r not found -- flying heuristic lead pursuit",
                        args.checkpoint)

    link = DCSLink(
        telemetry_port=args.telemetry_port,
        dcs_host=args.dcs_host,
        command_port=args.command_port,
    )
    ap = Autopilot(AutopilotConfig(invert_pitch=not args.no_invert_pitch))
    predictor = TurnRatePredictor()
    lead_predictor = TurnRatePredictor()
    state = PilotState(
        weapons_free=args.weapons,
        target_speed=args.target_speed,
        mode="formation" if args.formation else "engage",
        formation_station=args.formation or "combat_spread",
        formation_side=args.formation_side,
        leash=args.leash,
    )

    situation_lock = threading.Lock()
    latest_situation: dict = {"own": {}, "bandit": None, "lead": None}

    def get_situation() -> dict:
        with situation_lock:
            return copy.deepcopy(latest_situation)

    voice = None
    if args.radio:
        voice = start_radio(args, state, get_situation)

    # WSO back-seat advisory (human flies, AI calls the fight as text).
    wso = None
    wso_is_rule = False
    wso_lock = threading.Lock()
    latest_wso: dict = {}
    if args.wso:
        wso = make_wso(lang=args.wso_lang, model=args.wso_model, use_llm=args.wso_llm)
        wso_is_rule = isinstance(wso, WSOAdvisor)
        if not wso_is_rule:
            def wso_loop() -> None:
                while True:
                    time.sleep(max(0.5, args.wso_period))
                    with wso_lock:
                        sit = dict(latest_wso)
                    if not sit:
                        continue
                    try:
                        line = wso.advise(sit)
                    except Exception as exc:
                        print(f"WSO: agent error: {exc}")
                        continue
                    if line:
                        print(f"[WSO] {line}")
            threading.Thread(target=wso_loop, daemon=True, name="wso").start()
    advisory_only = args.wso and not args.wso_copilot

    log_writer = None
    log_file = None
    if args.log_csv:
        log_file = open(args.log_csv, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["t", "mode", "x_r", "y_r", "z_r", "x_b", "y_b", "z_b",
             "q_r", "q_b", "range", "gamma_cmd", "psi_cmd", "trigger"]
        )

    gamma_cmd, psi_cmd = 0.0, 0.0
    have_cmd = False
    last_decision = -1e9
    last_log = 0.0
    had_bandit = False
    called_guns = False
    in_formation_called = False

    log.info(
        "listening for DCS telemetry on udp/%d, sending commands to %s:%d "
        "(weapons %s, radio %s, %s%s",
        args.telemetry_port, args.dcs_host, args.command_port,
        "ENABLED" if args.weapons else "disabled",
        "on" if args.radio else "off",
        (f"WSO advisory [{args.wso_lang}], human flying, "
         if advisory_only else ("WSO advisory + AI flying, " if args.wso else "")),
        (f"formation {args.formation} on lead, leash {args.leash})"
         if args.formation else f"free engage, leash {args.leash})"),
    )
    try:
        while True:
            telem = link.receive()
            if telem is None:
                if time.time() - last_log > 5.0:
                    log.warning("waiting for telemetry... (is the mission running?)")
                    last_log = time.time()
                continue

            own = telem.own
            state.set_home(own.pos)
            g_own = own_gamma(own.tas, own.vv)
            if not have_cmd:
                gamma_cmd, psi_cmd = g_own, own.heading
                have_cmd = True

            trigger = 0
            feats = None
            act_b = None
            if telem.bandit is not None:
                predictor.update(telem.t, telem.bandit.pos)
                act_b = bandit_command_estimate(predictor, telem)
                feats = geo.situation(
                    list(own.pos), [own.tas, g_own, own.heading],
                    list(telem.bandit.pos), list(act_b),
                )
            if telem.lead is not None:
                lead_predictor.update(telem.t, telem.lead.pos)

            # ---- supervised autonomy: leash-gated commit / rejoin --------
            bandit_range = feats[2] if feats is not None else None
            if state.auto_commit(bandit_range is not None and bandit_range <= args.commit_range):
                if voice is not None:
                    voice.say("2, committing.")
                in_formation_called = False
            if state.auto_rejoin(bandit_range is None or bandit_range > args.rejoin_range):
                if voice is not None:
                    voice.say("2, rejoining.")

            # ---- radio: shared situation + proactive calls ---------------
            snap = state.snapshot()
            with situation_lock:
                latest_situation["own"] = {
                    "altitude_m": round(own.pos[2]),
                    "heading_deg": round(own.heading),
                    "speed_ms": round(own.tas),
                }
                latest_situation["mode"] = snap["mode"]
                latest_situation["weapons_free"] = snap["weapons_free"]
                latest_situation["leash"] = snap["leash"]
                latest_situation["formation"] = snap["formation_station"]
                latest_situation["lead"] = None
                if telem.lead is not None:
                    latest_situation["lead"] = {
                        "name": telem.lead.name,
                        "range_km": round(math.dist(own.pos, telem.lead.pos) / 1000.0, 1),
                    }
                latest_situation["bandit"] = None
                if telem.bandit is not None and feats is not None:
                    bearing = geo.wrap_heading(math.degrees(math.atan2(
                        telem.bandit.pos[0] - own.pos[0],
                        telem.bandit.pos[1] - own.pos[1],
                    )))
                    latest_situation["bandit"] = {
                        "name": telem.bandit.name,
                        "range_km": round(feats[2] / 1000.0, 1),
                        "bearing_deg": round(bearing),
                        "own_aspect_deg": round(feats[0]),
                        "altitude_m": round(telem.bandit.pos[2]),
                    }

            if voice is not None:
                if telem.bandit is not None and not had_bandit:
                    voice.say(f"2, tally {telem.bandit.name}, "
                              f"{math.dist(own.pos, telem.bandit.pos) / 1000.0:.0f} kilometers.")
                elif telem.bandit is None and had_bandit:
                    voice.say("2 is blind.")
                    called_guns = False
            had_bandit = telem.bandit is not None

            # ---- maneuver decision ---------------------------------------
            if snap["pending_break"]:
                state.resolve_break(own.heading)
                snap = state.snapshot()
            mode = snap["mode"]
            speed_cmd = snap["target_speed"]

            if mode == "vector" and snap["vector_heading"] is not None:
                psi_cmd = snap["vector_heading"]
                gamma_cmd = altitude_hold_gamma(snap["vector_altitude"], own.pos[2])
            elif mode == "anchor":
                if telem.t - last_decision >= args.decision_period:
                    last_decision = telem.t
                    psi_cmd = geo.wrap_heading(own.heading + 20.0)
                gamma_cmd = 0.0
            elif mode == "rtb":
                home = state.home
                if home is not None:
                    if math.dist(own.pos, home) < 2_000.0:
                        state.apply_anchor_arrival()
                    else:
                        gamma_cmd, psi_cmd = steer_to_point(own.pos, home)
            elif mode in ("formation", "scout"):
                lead_c = telem.lead
                if lead_c is not None:
                    if mode == "formation":
                        aim = form.station_position(
                            lead_c.pos, lead_c.heading,
                            snap["formation_station"], snap["formation_side"],
                        )
                        lead_spd = lead_speed_estimate(lead_predictor, lead_c, own.tas)
                        speed_cmd = form.formation_speed(
                            lead_spd, own.pos, aim, lead_c.heading)
                        if voice is not None and form.in_position(own.pos, aim):
                            if not in_formation_called:
                                voice.say("2, in formation.")
                                in_formation_called = True
                        else:
                            in_formation_called = False
                    else:  # scout
                        aim = form.scout_position(lead_c.pos, lead_c.heading)
                    gamma_cmd, psi_cmd = steer_to_point(own.pos, aim)
                else:
                    # no lead datalink yet: hold with a gentle right orbit
                    gamma_cmd = 0.0
                    psi_cmd = geo.wrap_heading(own.heading + 15.0)
            elif telem.bandit is None:
                # engage with no hostile airborne: gentle right orbit
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
                    gamma_cmd, psi_cmd = steer_to_point(own.pos, aim)

            # ---- weapons --------------------------------------------------
            if snap["weapons_free"] and mode == "engage" and feats is not None:
                if feats[2] < GUN_RANGE and feats[0] < GUN_ASPECT:
                    trigger = 1
                    if voice is not None and not called_guns:
                        voice.say("2, guns.")
                        called_guns = True

            # ---- WSO back-seat advisory (text) ---------------------------
            if wso is not None:
                wso_sit = build_wso_situation(
                    telem, own, g_own, feats, gamma_cmd, psi_cmd, mode, snap)
                if wso_is_rule:
                    line = wso.advise(wso_sit)
                    if line:
                        print(f"[WSO] {line}")
                else:
                    with wso_lock:
                        latest_wso.clear()
                        latest_wso.update(wso_sit)

            controls = ap.command(
                t=telem.t,
                pitch_deg=own.pitch,
                bank_deg=own.bank,
                heading_deg=own.heading,
                tas=own.tas,
                gamma_cmd_deg=gamma_cmd,
                psi_cmd_deg=psi_cmd,
                v_cmd=speed_cmd,
            )
            controls.trigger = trigger
            if not advisory_only:  # in WSO mode the human flies; never command
                link.send(controls)

            if log_writer is not None:
                b = telem.bandit
                log_writer.writerow([
                    round(telem.t, 3), mode, *[round(v, 1) for v in own.pos],
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
                    f"t={telem.t:8.1f}  mode={mode:<7}  tas={own.tas:5.1f}  "
                    f"alt={own.pos[2]:6.0f}  hdg={own.heading:5.1f}->{psi_cmd:5.1f}  "
                    f"gamma={g_own:5.1f}->{gamma_cmd:5.1f}  bandit: {bandit_txt}"
                    + ("  FIRING" if trigger else "")
                )
    except KeyboardInterrupt:
        log.info("AI pilot stopped")
    finally:
        if voice is not None:
            voice.close()
        if log_file is not None:
            log_file.close()
        link.close()


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version",
                   version=f"UCAV AI Pilot {__version__}")
    p.add_argument("--license-key", default=None,
                   help="Pro license key (else $UCAV_LICENSE_KEY or "
                        "~/.ucav_pilot/license.key; unset = trial)")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="load options from a TOML config file (CLI flags override)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="logging verbosity")
    p.add_argument("--log-file", default=DEFAULT_LOG_FILE, metavar="FILE",
                   help="rotating log file (empty string = console only)")
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

    cca = p.add_argument_group("CCA loyal-wingman teaming (MUM-T)")
    cca.add_argument("--formation", nargs="?", const=form.DEFAULT_FORMATION,
                     default=None, choices=sorted(form.FORMATIONS),
                     help="start holding this formation on the crewed flight "
                          "lead instead of free engaging (bare flag = "
                          f"{form.DEFAULT_FORMATION})")
    cca.add_argument("--formation-side", default="right", choices=["left", "right"],
                     help="which side of the lead to hold station on")
    cca.add_argument("--leash", default="tight", choices=["close", "tight", "loose"],
                     help="supervised-autonomy level: close=formation only, "
                          "tight=auto-commit but weapons held, "
                          "loose=auto-commit weapons free")
    cca.add_argument("--commit-range", type=float, default=15_000.0,
                     help="m; leave formation to engage a bandit inside this range")
    cca.add_argument("--rejoin-range", type=float, default=25_000.0,
                     help="m; rejoin formation when the bandit is beyond this range")

    wsog = p.add_argument_group("WSO back-seat advisory (text)")
    wsog.add_argument("--wso", action="store_true",
                      help="AI rides in back as WSO and advises the human pilot "
                           "as text; by default it does NOT fly the jet")
    wsog.add_argument("--wso-lang", default="ko", choices=["ko", "en"],
                      help="advisory language (Korean/English)")
    wsog.add_argument("--wso-llm", action="store_true",
                      help="use a Claude back-seater for free-form advice "
                           "(default: offline rule-based advisor)")
    wsog.add_argument("--wso-model", default="claude-opus-4-8",
                      help="Claude model for the LLM back-seater")
    wsog.add_argument("--wso-period", type=float, default=4.0,
                      help="seconds between LLM back-seater calls")
    wsog.add_argument("--wso-copilot", action="store_true",
                      help="let the AI fly while also advising "
                           "(default: human flies, AI is advisory only)")

    radio = p.add_argument_group("radio wingman (LLM voice commands)")
    radio.add_argument("--radio", action="store_true",
                       help="enable the voice/text radio wingman")
    radio.add_argument("--radio-model", default="claude-opus-4-8",
                       help="Claude model for the radio agent")
    radio.add_argument("--radio-lang", default="ko-KR",
                       help="speech recognition language (e.g. ko-KR, en-US)")
    radio.add_argument("--radio-offline", action="store_true",
                       help="skip the LLM and use the rule-based brevity parser")
    radio.add_argument("--radio-text-only", action="store_true",
                       help="console text radio (no microphone / TTS)")
    return p


if __name__ == "__main__":
    main()
