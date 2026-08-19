# air-combat-Reinforcement-Learning

![CI](https://github.com/godlovetco/air-combat-Reinforcement-Learning/actions/workflows/ci.yml/badge.svg)

UCAV air-combat maneuver decision by reinforcement learning — now with a
**DCS World addon** that lets the trained AI pilot fly a real aircraft inside
Digital Combat Simulator, team with a crewed lead as a CCA loyal wingman, or
ride in back as a **WSO** that advises a human pilot as text.

```
 flight lead voice ──STT──▶ LLM radio wingman (Claude) ──orders──┐
        ▲                                                        ▼
       TTS ◀──radio replies (brevity calls)──┐   ┌── tactical order state
                                             │   │  (engage/anchor/vector/break/
                                             │   │   RTB/formation/scout/leash)
DCS World ──UDP telemetry──▶ dcs_bridge.run_pilot ──UDP stick/throttle──▶ DCS World
 (Lua Export script)              │
 own + bandit + lead datalink     ├─ CCA loyal-wingman teaming (formation on lead)
                                  ├─ target trajectory prediction (lead pursuit)
                                  ├─ Q-network maneuver decision (9 actions, 2 Hz)
                                  └─ bank-to-turn autopilot (20 Hz)
```

The maneuver model is the original one from this repository: the agent picks
one of **9 discrete maneuvers** (climb angle ±10°/hold × heading ±10°/hold)
from a value network whose input is the 72-dim "situation that each maneuver
would lead to" vector. The legacy TensorFlow 1.x code is preserved in
`main.py` / `class_env.py`; everything new is pure Python 3 + numpy.

## AI pilot validation in Digital Combat Simulator

The DCS integration follows the validation methodology of:

> J. Yoo, D. Kim, D. H. Shim, *"Deep Reinforcement Learning based Autonomous
> Air-to-Air Combat using Target Trajectory Prediction,"* ICCAS 2021 —
> which validated a reinforcement-learning AI pilot in DCS World via a Lua
> socket link running at ~20 Hz, and showed that predicting the target's
> trajectory (10 future positions over a 4 s window) is key to converting a
> chase into an intercept.

What this addon reproduces from the paper, and where it differs:

| Paper (Yoo/Kim/Shim 2021)                  | This addon                                             |
|--------------------------------------------|--------------------------------------------------------|
| Lua TCP socket link to the RL code, ~20 Hz | Lua **UDP** link at 20 Hz (latest-packet-wins, no stalls) |
| PPO + LSTM policy                          | Q-network over the repo's legacy 9-maneuver action set |
| Seq2Seq-LSTM target trajectory prediction   | Constant-turn-rate predictor, 10 samples over 4 s      |
| Aspect-angle + distance reward shaping      | Same shaping terms in `dcs_bridge/sim_env.py`          |
| Random-spawn enemy missions for validation  | Same recommended mission setup (see below)             |
| Post-flight tracking analysis               | `--log-csv` per-tick engagement log for plots          |

## Installation (DCS side, Windows)

**One-click:** install Python 3.9+ (`pip install numpy`), then double-click
**`install.bat`**. It copies the Lua addon into your DCS `Saved Games\Scripts`
folder and registers it in `Export.lua` for you — idempotently, chaining any
exports you already run (Tacview, SRS, DCS-BIOS…), so nothing you had breaks.

```
install.bat                      # auto-detects DCS / DCS.openbeta and installs
python -m dcs_bridge.install --dry-run     # preview without changing anything
python -m dcs_bridge.install --uninstall   # clean removal
python -m dcs_bridge.install --saved-games "D:\Saved Games\DCS.openbeta"
```

Ports: telemetry `udp/7778` out, commands `udp/7779` in, localhost only by
default (edit the constants at the top of `UCAVPilotExport.lua` to change).

**Python side (console commands):** installing the package gives you the
`ucav-pilot`, `ucav-install`, `ucav-train`, and `ucav-license` commands:

```
pip install .                # core (numpy only)
pip install ".[all]"         # + Pro radio/voice/plots dependencies
ucav-install                 # same as install.bat
ucav-pilot --radio --license-key UCAV1.xxxx.yyyy
```

Optional-dependency extras: `radio` (Claude), `voice` (STT/TTS), `plots`
(matplotlib), or `all`.

**Config file (no flags to memorize):** copy `ucav_pilot.example.toml`, edit,
and pass `--config`. Any command-line flag still overrides the file.

```
ucav-pilot --config ucav_pilot.toml            # options come from the file
ucav-pilot --config ucav_pilot.toml --weapons  # ...but flags win
```

**Logging:** lifecycle and error messages are logged to the console and to a
rotating file at `~/.ucav_pilot/logs/ucav_pilot.log` (attach it when reporting
an issue). Tune with `--log-level DEBUG|INFO|WARNING|ERROR` and `--log-file`
(empty string = console only).

**Reporting a problem:** run `ucav-support` to collect a diagnostics bundle
(version, OS/Python, license *status*, DCS addon install state, checkpoint
fingerprint, log tail) into a zip you can attach to a support request. It
contains **no license key and no API key** — only whether they are set — and
everything in it is plain text, so you can inspect the bundle before sending
it. `ucav-support --print` shows the same data without writing a file.

<details><summary>Manual install (if you prefer)</summary>

1. Copy `dcs-addon\Scripts\UCAVPilot\` into
   `%USERPROFILE%\Saved Games\DCS\Scripts\`.
2. If you have **no** `Saved Games\DCS\Scripts\Export.lua`: copy
   `dcs-addon\Export.lua` there. If you already have one: append the loader
   block from `dcs-addon\Export.lua` to the end of it.
</details>

## Editions & licensing

UCAV AI Pilot is a commercial product (see `LICENSE`). It runs out of the box
in **Trial**; the **Pro** tier unlocks the Claude-powered features.

| Feature | Trial (no key) | Pro (licensed) |
|---|:---:|:---:|
| RL air-combat pilot + autopilot | ✅ | ✅ |
| CCA loyal-wingman teaming (formation / leash) | ✅ | ✅ |
| Offline rule-based radio & WSO (KOR/ENG) | ✅ | ✅ |
| **LLM radio wingman** (Claude, free-form voice) | — | ✅ |
| **LLM WSO back-seater** (Claude commentary) | — | ✅ |

Without a key the product never refuses to run — it simply falls back to the
offline rule-based tier, so a trial is always usable. Supply a Pro key with
`--license-key`, the `UCAV_LICENSE_KEY` environment variable, or a
`~/.ucav_pilot/license.key` file:

```
python -m dcs_bridge.run_pilot --radio --license-key UCAV1.xxxx.yyyy
python -m dcs_bridge.licensing verify UCAV1.xxxx.yyyy      # check a key
```

Keys are offline-verifiable signed tokens; the vendor mints them with
`python -m dcs_bridge.licensing issue …` (see `dcs_bridge/licensing.py` for
the security model and the production hardening path).

## Flying

1. Create a mission with a **player/client slot** in the aircraft the AI
   should fly (the export API controls the player's aircraft), plus one or
   more hostile AI fighters. Following the paper's validation setup: F-16,
   ~20,000 ft, ~230 kn, enemy spawned at random positions/headings.
2. Start the mission, enter the cockpit, get airborne (or use an air start).
3. On the same machine run:

   ```
   python -m dcs_bridge.run_pilot --checkpoint checkpoints/ucav_policy.npz --log-csv flight_01.csv
   ```

4. Hands off — the AI pilot takes the stick: it finds the nearest hostile
   aircraft, predicts its trajectory, and maneuvers using the trained policy.

Useful options:

| Option | Effect |
|---|---|
| `--heuristic` | fly lead pursuit only (no network) — use this first to tune autopilot gains |
| `--weapons` | allow trigger pulls inside the gun envelope (default: off) |
| `--decision-period 0.5` | seconds between policy decisions |
| `--target-speed 250` | TAS in m/s the throttle loop holds |
| `--log-csv out.csv` | per-tick validation log (positions, aspect angles, range) |

After a flight, analyze the log the way the paper does its validation:

```
python -m dcs_bridge.flight_report flight_01.csv --plot flight_01.png
```

prints engagement statistics (min/mean range, tracking fraction, time to
first gun solution, mode timeline) and renders trajectory / range / aspect /
altitude plots (matplotlib required only for `--plot`).
| `--no-invert-pitch` | flip the pitch-axis sign if the jet pushes instead of pulls |

Safety defaults: weapons release is **off** unless `--weapons` is passed (or
the flight lead radios "weapons free"), and the Lua script neutralizes the
controls and returns the jet to you if the Python agent stops sending
commands for 1 second.

## LLM radio wingman (voice commands / 음성 명령)

`--radio` puts an LLM-based AI pilot on the net as your wingman ("Viper 2"):
speak (or type) commands in Korean or English, and it interprets them
against the live tactical picture, executes the order, and answers with a
short brevity call over TTS. It also makes proactive calls — "tally",
"guns", "blind".

```
pip install anthropic                # + optionally: SpeechRecognition pyaudio pyttsx3
set ANTHROPIC_API_KEY=sk-ant-...     # your Claude API key
python -m dcs_bridge.run_pilot --radio --checkpoint checkpoints/ucav_policy.npz
```

Example net traffic:

```
lead : "2번기, 우측 브레이크!"          → hard right turn, "2, breaking right!"
lead : "vector heading 270 angels 15"  → flies 270 at 15,000 ft
lead : "weapons free"                  → trigger allowed in the gun envelope
lead : "2, say status"                 → "2 has tally, MiG-29 at 7.5 km, ..."
lead : "기지로 복귀"                    → RTB to the start point, then anchors
```

How it works: the transmission plus a live `<situation>` block (own state,
bandit bearing/range/aspect, mode, weapons state) goes to Claude
(`claude-opus-4-8` by default, adaptive thinking at low effort for radio
latency). The model issues orders through a strict `set_order` tool —
engage / anchor / vector / break left-right / RTB / weapons free-hold —
which the 20 Hz flight loop executes, and its text reply is spoken back.

| Option | Effect |
|---|---|
| `--radio-model claude-opus-4-8` | which Claude model answers the radio |
| `--radio-lang ko-KR` | speech-recognition language (`en-US`, ...) |
| `--radio-text-only` | console text radio: no microphone or TTS needed |
| `--radio-offline` | no API: rule-based brevity parser (KOR/ENG) handles the standard commands |

Without an `ANTHROPIC_API_KEY` the radio automatically drops to the offline
brevity parser, so voice command always works — the LLM adds free-form
understanding ("bandit's behind you, get out of there and come home" still
becomes *break* + *RTB*) and situation-aware replies.

## CCA loyal-wingman teaming (MUM-T)

Modeled on the U.S. Air Force **Collaborative Combat Aircraft (CCA)**
program: instead of free-engaging alone, the UCAV flies as an uncrewed
"loyal wingman" on a **crewed flight lead**, holding tactical formation and
committing on threats under *supervised autonomy* — the human sets how long
the leash is, the drone flies the details.

The Lua export reports the nearest friendly aircraft as the flight lead
(`lead` datalink object); the flight loop stations the UCAV on it and paces
its speed to hold position.

```
python -m dcs_bridge.run_pilot --formation combat_spread --leash tight \
       --radio --checkpoint checkpoints/ucav_policy.npz
```

**Formations** (`--formation`, or radio "combat spread", "line abreast",
"fighting wing", "wedge", "echelon", "wall", "trail"; `--formation-side
left|right`) are the standard two-ship geometries relative to the lead's
nose. **Scout** (radio "push"/"scout") sends the UCAV ahead of the lead to
sweep.

**Leash** (`--leash`, or radio "close/tight/loose leash", "weapons tight")
is the supervised-autonomy level:

| Leash | Behavior |
|---|---|
| `close` | station-keeping only; weapons caged, never leaves formation |
| `tight` (default) | auto-commits on a bandit inside `--commit-range` (15 km) but holds fire until the lead calls weapons free |
| `loose` | auto-commits **and** fires in the gun envelope on its own |

When the threat opens beyond `--rejoin-range` (25 km) the UCAV rejoins
formation on its own. Proactive radio calls mark the transitions:
"2, in formation" / "2, committing" / "2, rejoining".

```
lead : "2, combat spread, right side"   → stations abeam-high on the right
lead : "you're tight"                    → may commit, weapons stay caged
lead : "2, weapons free"                 → cleared to fire on its commit
lead : "push ahead and scout"            → runs 5 km in front to sweep
lead : "편대 복귀"                        → rejoins formation
```

## WSO back-seat advisory (AI advises a human pilot / 텍스트 조언)

`--wso` flips the roles: **you fly, the AI rides in back as the Weapon
Systems Officer** and calls the fight as text over the intercom. It never
touches the controls — it only advises.

```
python -m dcs_bridge.run_pilot --wso --wso-lang ko --checkpoint checkpoints/ucav_policy.npz
```

What the back-seater calls, prioritized so the urgent stuff never waits:

| Priority | Call | Example |
|---|---|---|
| Safety | ground/altitude, hard defensive | "고도! 고도! 기수 당겨!" / "Break now, chaff and flares!" |
| Threat | spike, bandit nose-on and closing | "스파이크, 4킬로. 브레이크 준비." |
| Weapons | gun parameters / in-range cue | "건 사정권, 파라미터 안. 사격!" |
| Energy | corner speed, low-and-slow | "코너 속도 초과, 당겨서 선회율 높여." |
| Geometry | overshoot / high yo-yo | "오버슈트 주의, 하이 요요로 에너지 살려." |
| **Maneuver** | **the RL policy's recommended move, in plain words** | "추천: 우로 210, 기수 올려." / "Recommend come right to 210, nose up." |
| Picture | contact / BRA / no-joy | "브라 045, 8킬로, 고도 6000미터." |

The signature feature is the **maneuver recommendation**: the same trained
Q-network that can fly the jet instead whispers what *it* would do, so the
human pilot gets the model's decision as advice rather than as stick input.

| Option | Effect |
|---|---|
| `--wso-lang ko` | advisory language (`ko` / `en`) |
| `--wso-llm` | use a Claude back-seater for free-form advice (default: offline rule-based) |
| `--wso-period 4` | seconds between LLM back-seater calls |
| `--wso-copilot` | let the AI fly *and* advise (default: human flies, AI advisory only) |

The rule-based advisor is offline and runs inside the 20 Hz loop, so
time-critical calls are never late; `--wso-llm` adds a Claude back-seater on
a background timer for richer, situation-aware phrasing (falls back to the
rule-based advisor without an `ANTHROPIC_API_KEY`).

## Training

```
pip install -r requirements.txt
python -m dcs_bridge.train --episodes 1000 --out checkpoints/ucav_policy.npz
python -m unittest discover -s tests     # test suite
```

Training runs the point-mass 1v1 engagement (head-on merge at 10 km,
250 m/s, win = inside 2,500 m with <30° own aspect, >30° bandit aspect and
an altitude advantage) with the legacy bugs fixed: proper degree→radian
conversion in the bandit aspect angle, no re-initialization of the network
between updates, plus experience replay, a target network, an ε-greedy
schedule and best-checkpoint selection (periodic greedy evaluation decides
which weights are kept).

**Reactive opponents** (`--opponent`): the bandit is no longer only
straight-flying. `pursuit` turns to point at the agent (a turning fight),
`evasive` breaks toward the beam when threatened from behind, `ace`
switches between the two by who currently holds the angular advantage, and
`mixed` randomizes the behavior per episode so the policy learns to
generalize instead of overfitting one target profile. `--eval-opponent`
picks the behavior the periodic/final greedy evaluation scores against, and
`--mixed-weights "pursuit=5,straight=1,evasive=1"` biases the per-episode
draw — a rehearsal curriculum that trains one behavior hard while
rehearsing the others, so `--init` fine-tuning doesn't forget them.

**Self-play** (`--opponent selfplay`) drops the scripted intents entirely
and flies the bandit with a *frozen* policy checkpoint, chosen with
`--selfplay-init` (default: a frozen copy of the learner's starting
weights, i.e. "beat the current champion"). `--selfplay-refresh N` promotes
the learner to be its own opponent every N episodes. The opponent is always
frozen inside an episode — it never trains mid-fight — so the Q-targets stay
on a stationary problem. Self-play is *not* drawn by a plain `--opponent
mixed`; weight it explicitly (`--mixed-weights "selfplay=3,pursuit=1,..."`)
to fold it into a rehearsal curriculum.

**Energy action set** (`--action-set energy`) is the fidelity upgrade the
self-play numbers argued for. The original 9 maneuvers hold speed fixed, so
every turn is free and a fight between equals decays into pure angles —
which is why a mirror match ended `out_of_bounds` 92% of the time rather
than resolving. The energy set crosses those 9 with a throttle axis
(burner / hold / idle) for **27 actions**, and couples speed to the flight
path:

```
dv/dt = throttle x 4.0 m/s^2  -  g sin(gamma)  -  3.0 m/s^2 x (turn / 10 deg)
```

Burner buys speed, climbing spends it, and a hard turn bleeds it — a 30°
climb costs 4.9 m/s² against the 4.0 m/s² the engine buys, so you cannot
climb and accelerate at once. Speed is clamped to 120–400 m/s. That is the
energy-maneuverability trade the legacy action set had no way to express.

The two action sets are separate model families: `energy` networks are
216→27 instead of 72→9, so their checkpoints are **not interchangeable**
(`--init` refuses a mismatch, and `run_pilot` reads the action set off the
checkpoint's tensor shapes and drives the throttle loop from the policy's
choice). `legacy` remains the default and is bit-for-bit unchanged; scripted
bandits other than `straight` fly the same energy model at full throttle, so
the fight stays fair, while `straight` keeps constant speed because constant
everything *is* the classic profile. Energy-mode scores are therefore not
comparable to legacy-mode scores.

Two things had to be fixed for the energy set to be trainable at all, both
worth knowing if you extend the feature vector:

* **`delta_v2` was effectively unnormalized.** The legacy scaling divides the
  situation vector by `[200, 200, 20000, 200, 10000, 1, 40000, 10000]` — note
  the `1` on `v_r² − v_b²`. That was harmless only because the legacy action
  set pins both aircraft at 250 m/s, so the feature is *identically zero* in
  every situation it ever sees. Let speed vary and it reaches 145,000 while
  every other normalized feature is order 1, swamping the first layer. The
  energy set scales it like `v2`; the legacy constants are untouched.
* **From scratch, 216→27 is a much harder problem.** `--transfer-init` lifts a
  trained legacy policy into the energy family: energy candidate `3i+k` is
  legacy candidate `i` at burner/hold/idle, so each input block is copied into
  all three slots (scaled 1/3) and each output column replicated three times.
  The result agrees with the legacy policy's maneuver choice ~97% of the time
  and starts the fine-tune with only the throttle axis to learn.

A trained energy policy ships as `checkpoints/ucav_policy_energy.npz`
(`--checkpoint checkpoints/ucav_policy_energy.npz`). It was built by lifting
the legacy policy with `--transfer-init` and fine-tuning through the rehearsal
curriculum with self-play weighted in, **under the corrected climb physics**.
400 engagements per behavior on a held-out seed; the self-play row is scored
against the previous energy checkpoint:

| Bandit behavior | previous checkpoint | **shipped now** |
|---|---|---|
| straight | 0.98 | 0.93 |
| pursuit | 0.92 | **0.94** |
| evasive | 0.98 | 0.94 |
| ace | 0.99 | 0.96 |
| mixed | 0.98 | 0.93 |
| **self-play** | 0.35 (142 W / 236 L) | **0.81 (322 W / 67 L)** |

This one is a deliberate exception to the no-regressions rule this project
otherwise holds to, and the reason is specific rather than a lowered bar: the
previous checkpoint was trained before the zoom-climb exploit was found, on a
model where an aircraft could hold 70° of climb on the speed floor forever. Its
scripted scores are real, but they were earned in a world the simulator no
longer has. The replacement costs about four points against scripted bandits
and turns the fight against an equal from losing better than 3:2 into winning
nearly 5:1.

Two further curriculum stages were run to try to buy those four points back —
tripling the straight rehearsal, then re-weighting toward the scripted
behaviors — and both came out worse overall (0.89 / 0.85 on straight and
pursuit). The trade appears to be real rather than a tuning miss.

**A hypothesis, falsified, and what it turned up.** The energy action set was
added on the theory that a mirror match cannot resolve without an energy game.
It helped — unresolved mirror fights fell from 92% to 60% — but a majority
still ended `out_of_bounds`, so the README said the remaining cause was
probably the 200 km arena. That was wrong, and easy to check: widening the
arena to 2,000 km and quadrupling the episode cap changed the outcome
distribution by *exactly zero episodes*. Breaking the departures down by which
boundary was crossed said why —

```
146  bandit hit the 11 km ceiling
 29  agent hit the ceiling
  5  both
  0  horizontal boundary       0  ground
```

Every unresolved fight was a zoom climb into the lid, and the *bandit* was
departing five times more often than the agent. Two defects behind it:

* **A hard speed floor makes a vertical climb free.** Clamping speed at
  `V_MIN` let a point mass trade all its energy for altitude, bottom out, and
  keep climbing at 120 m/s forever. Real jets run out and the nose falls, so
  climb authority now scales from the full 70° at corner speed (200 m/s) down
  to level flight at `V_MIN` (`geometry.max_climb_angle`). Descending is never
  limited — unloading is how you get the energy back. The jet still climbs when
  it has thrust to spare; it settles at the angle its energy sustains instead
  of holding 70° on the floor.
* **The agent was penalized for the bandit's departure.** `out_of_bounds` cost
  −5 whenever *either* aircraft left the box, so in a self-play fight the agent
  ate the penalty for something it does not control, five times out of six. The
  environment now attributes it: `out_of_bounds` (−5) when the agent leaves,
  `bandit_departed` (0.0) when only the bandit does. This applies to both action
  sets; evaluation metrics are unaffected because neither outcome counts as a
  win or a conversion.

Running the shipped energy policy unchanged under the corrected physics:

| | before | after |
|---|---|---|
| ceiling / boundary departures | 60% | ~1% |
| timeouts | 0% | 40% |
| **resolved (win or loss)** | **40%** | **60%** |

The remaining 40% are genuine stalemates — two equally matched fighters in a
turning fight that neither converts — which is a real outcome rather than an
artifact of the box.

## Beyond-visual-range engagement (BVR)

Everything above models a gun fight: close to 10 km, convert to the bandit's
six, take the shot. A BVR fight is a different problem, decided long before
anyone points at anyone — by radar, weapon envelope and missile timeline.
`dcs_bridge/bvr.py` is the physics and `dcs_bridge/bvr_env.py` is the arena;
`--engagement bvr` trains against it.

**Radar.** Two array types, differing in the four places that change how the
fight is flown:

| | mechanically scanned | AESA |
|---|---|---|
| detection range | 90 km | 135 km |
| Doppler notch gate | 50 m/s | 30 m/s |
| re-acquisition after a lost track | 3.0 s | 0.5 s |
| missiles guided at once | 1 | 4 |
| heard by the target's RWR at | 180 km | 68 km |

That last row is the tactical argument for the technology. A warning receiver
only listens one way while a radar needs the return trip, so a conventional
set announces itself at twice its own detection range — *spiked before seen* is
the normal state of affairs. An LPI array inverts it and can hold a track from
outside the range its emissions register at.

The notch is modeled on the **target's** radial velocity, not on total closure.
This matters: a target beaming the antenna disappears into the ground clutter
no matter how fast the shooter is closing, and modeling it off closure would
have made notching impossible against a fast shooter — backwards.

**Weapons.** Three, because they impose different fights:

| | reach (head-on, 9 km) | no-escape zone | notes |
|---|---|---|---|
| `ARH-medium` | 64 km | 23 km | the workhorse |
| `ARH-long` | 110 km | 55 km | ramjet; the NEZ is large enough that turning and running stops working |
| `IR-short` | 17 km | 7 km | fire-and-forget, immune to notching, has to be carried to the merge |

`WeaponEngagementZone` carries Rmax / **Rtr** / Rne / Rmin and time of flight.
Rtr — the reach against a target that reverses the instant you shoot — is the
band that decides whether a shot is worth taking at all.

**Missiles in flight** are inertial on the launcher's radar until the seeker
goes active, then self-guiding. Three ways to defeat one, all tested: break the
supporting lock before pitbull, hold the beam on it after, or outrun its energy.
A notched seeker is *not* dead — it coasts on its last solution for
`seeker_memory` seconds and reacquires if the target leaves the notch, so a
notch is a timing problem rather than a switch.

**The arena** starts 70 km apart head-on, carries per-side radars and loadouts
so an AESA-versus-mechanical matchup is expressible, and ends on a missile hit,
a merge (both sides Winchester inside 9 km), a departure or a timeout.

*When* to shoot is a rule — locked, inside the envelope, prefer the no-escape
zone, respect trigger discipline and the array's track capacity — because that
part is well understood. *How to fly* is what the policy learns, and the
observation carries a 15-wide block for it: locks, RWR spike, rounds remaining
on both sides, what is in the air, support still owed, envelope flags, and the
threat's bearing and notch depth.

### Two results worth reading before trusting the numbers

**An AESA is worth very little in a symmetric 1v1.** Sweeping how precisely the
bandit beams, the same hand-flown agent scores 0.05 with a mechanical set and
0.06 with an AESA; at a sloppier notch, 0.14 against 0.15. The reason is
structural rather than a tuning miss: what defeats a shot is the *missile
seeker's* notch gate, and a seeker does not inherit the launching aircraft's
array. The edge that does appear grows with start range, which is where the
extra detection range can be spent. Reported as measured rather than tuned
until it looked better.

**The BVR pilot is scripted, and that is a result, not a placeholder.**
`dcs_bridge/bvr_pilot.py` flies the timeline: defend a guiding missile by
beaming *it* (not the jet that fired it — those bearings diverge as the round
closes), support your own shot by cranking as far off the target as the antenna
tolerates, otherwise commit. Twenty lines, three strictly-ordered states.

Every reinforcement-learning policy trained against this environment lost to
it. Win rate / survival rate over **400 engagements per cell on each of two
held-out seeds**:

| agent | straight | pursuit | evasive | ace |
|---|---|---|---|---|
| uniform random actions | 0.09/0.87 | 0.21/0.93 | 0.00/0.83 | 0.00/0.94 |
| always hot (fly at the bandit) | 0.31/0.31 | 0.54/0.54 | 0.00/0.07 | 0.00/0.99 |
| **`TimelinePilot` (shipped)** | **0.96**/1.00 | **1.00**/1.00 | **0.13**/0.98 | 0.00/1.00 |
| best RL policy | 0.61/0.88 | 0.80/0.86 | 0.09/0.64 | 0.01/0.98 |

The two floors are worth as much as the ceiling. Uniform random actions almost
never score but survive 83–94% of the time — a target that maneuvers
unpredictably is hard to lock — so survival read on its own is a cheap number.
Flying straight at the bandit converts 0.31 and 0.54 but survives 0.31 and
0.07: pressing without defending is how you die. A useful policy has to beat
both, and the learned ones only just do.

Getting the RL policy even that far took two fixes, both of which were bugs
rather than tuning:

* **The observation did not locate the threat.** It announced that a missile
  was inbound and how long it had, and never said *where* it was — so a
  maneuver defined relative to the threat was unlearnable. Adding bearing and
  notch depth moved the straight column 0.07 → 0.19.
* **Checkpoints were selected against `ace`, where nothing scores** — the
  scripted pilot included. That left survival as almost the entire selection
  signal, which rewards hiding. Selecting on the training mixture instead moved
  straight 0.34 → 0.65 and pursuit 0.58 → 0.79 with no other change.

Tripling the training budget after that made things *worse* (straight 0.65 →
0.51, and survival against `ace` 1.00 → 0.68), which is the usual DQN story and
not something more episodes will fix.

So the scripted pilot ships and the learned one does not. An earlier revision
of this section claimed the RL policy had overtaken the timeline on the
`evasive` and `ace` columns; that was a 150-episode sample and did not survive
being re-run at 400.

Note also that the `evasive` and `ace` columns are near zero *for every agent
including the scripted one*: two pilots who both notch correctly tend to end
BVR at the merge. That is a real dynamic rather than an artifact, and it is why
`merge` is its own outcome rather than being scored as a draw-shaped win.

## Repository layout

```
dcs-addon/                Lua export addon for DCS World
  Export.lua                loader stub for Saved Games\DCS\Scripts\
  Scripts/UCAVPilot/        the export script (telemetry out, commands in)
dcs_bridge/               Python package (numpy; anthropic for the radio)
  bvr.py                    BVR physics: radar/notch, weapon envelope, missiles
  bvr_env.py                BVR 1v1 arena (70 km start, missile timeline)
  bvr_pilot.py              scripted BVR pilot (defend / support / commit)
  geometry.py               combat geometry & the legacy 72-dim network input
  policy.py                 Q-network (72→100→30→9), training + inference
  predictor.py              constant-turn-rate target trajectory prediction
  formation.py              CCA loyal-wingman formation station-keeping (MUM-T)
  sim_env.py                point-mass 1v1 training environment
  autopilot.py              bank-to-turn inner loop (maneuver → stick axes)
  link.py                   UDP protocol to/from the Lua script
  orders.py                 tactical orders + thread-safe pilot state
  wingman.py                LLM radio agent (Claude) + offline brevity parser
  wso.py                    WSO back-seat text advisor (rule-based + LLM)
  voice.py                  STT/TTS with console fallback
  licensing.py              Pro license-key verification (gates the LLM tier)
  install.py                one-click DCS addon installer (python -m dcs_bridge.install)
  config.py                 TOML config file loader (--config)
  logging_setup.py          console + rotating-file logging
  support.py                diagnostics bundle for support (ucav-support)
  flight_report.py          post-flight validation stats and plots
  train.py                  python -m dcs_bridge.train
  run_pilot.py              python -m dcs_bridge.run_pilot
checkpoints/              trained policy weights (.npz)
tests/                    unit tests (python -m unittest discover -s tests)
install.bat               Windows one-click installer wrapper
ucav_pilot.example.toml   sample --config file
pyproject.toml            packaging: pip install, console scripts, extras
.github/workflows/        CI (test matrix + build) and tagged-release automation
LICENSE                   commercial EULA
main.py, class_env.py     original TF 1.x project (kept as-is)
```

---

## 原始项目说明 (original notes)

利用值函数逼近网络设计无人机空战自主决策系统，采用epsilon贪婪策略，三层网络结构。
其中包含了无人机作为质点时的运动模型和动力学模型的建模。
由于无人机作战的动作是连续并且复杂的，本项目仅考虑俯仰角gamma(又叫航倾角)和航向角pusin的变化，并且离散的规定每次变化的幅度为10度，假定速度v为恒定值。根据飞机的运动模型，由俯仰角、航向角和速度可以推算出飞机位置的改变，即x,y,z三个方向的速度分量，在每一步中，根据这些分量变化位置position信息，posintion中的三个值为x,y,z坐标，是东北天坐标系下的坐标值。从坐标信息和角度信息以及速度信息，可以计算出两个飞机的相对作战态势state。
在上文中提到，我们的动作是仅对俯仰角和航向角进行改变，即增大，减少和不变，故两个角度的变化组合一共有3×3=9种动作。在每个态势下，都有9种动作可以选择，将这个态势下的9种动作将会产生的新的态势，作为网络的输入，网络的输出是9个数字，代表每个动作的值函数。
由于是无监督学习，故我们需要利用值函数的Bellman公式生成标签。本文利用时间差分思想。
怎么生成标签呢？我们先进行一次完整的游戏，将这次游戏中所有的值都记录下来，然后在游戏结束之后，根据胜利（红方为我方）、失败和最大步数、超出范围等给出一个回报值，然后设定一个回报的衰减率，将这个回报值根据步数越靠前影响越小的规律添加在每一步的目标中。
胜利的标准是，在攻击距离内达到攻击角度。
在训练后的结果：在对方匀速直线运动向我方飞行的时候，我方先快速爬升到一定高度，然后俯冲向敌机。
class_env文件是环境类，包含了无人机建模和各种坐标角度与态势之间的转换。
main文件是主文件，有网络和主要的执行过程和测试过程（TensorFlow 1.x / Python 3.6，仅作存档保留；新的训练入口是 `python -m dcs_bridge.train`）。
