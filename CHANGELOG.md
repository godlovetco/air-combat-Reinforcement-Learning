# Changelog

All notable changes to UCAV AI Pilot are documented here. This project aims
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Beyond-visual-range engagement** (`--engagement bvr`): a second engagement
  type with its own physics module and arena.
  - `bvr.py` — typed radars (mechanically-scanned vs **AESA**, differing in
    detection range, Doppler notch width, re-acquisition time, simultaneous
    tracks, and the RWR asymmetry that makes LPI worth having); a **weapon
    engagement zone** with Rmax/Rtr/Rne/Rmin and time of flight; a **missile
    catalog** (medium ARH, ramjet ARH, IR) and missiles in flight that are
    inertial on the launcher's radar until pitbull, coast on seeker memory when
    notched, and can be defeated by breaking the mid-course lock, holding the
    beam, or outrunning their energy.
  - `bvr_env.py` — 70 km head-on arena with per-side radars and loadouts,
    rule-based shot selection, an `ace` bandit that flies the timeline off its
    RWR, and a 15-wide observation block (locks, spike, rounds remaining,
    what is in the air, support owed, envelope flags, threat bearing and notch
    depth). `evaluate` reports **survival** rather than conversion here.
  - `bvr_pilot.py` — **`TimelinePilot`**, the scripted BVR pilot that ships:
    defend a guiding missile by beaming it, support your own shot by cranking,
    otherwise commit. Over 400 engagements per cell on two held-out seeds it
    scores 0.96 / 1.00 / 0.13 / 0.00 against straight / pursuit / evasive / ace
    with survival 1.00, 1.00, 0.98, 1.00 — beating every RL policy trained
    against the environment on every axis.
  - **WSO BVR calls** — the back-seater now calls the missile fight: spike,
    defend with the threat's clock position and the shorter notch direction,
    support countdown, pitbull, shoot / hold-for-the-NEZ, winchester.
  - Measured, and reported as measured: an AESA is worth very little in a
    symmetric 1v1 (0.05 vs 0.06, 0.14 vs 0.15 across a notch-precision sweep),
    because what defeats a shot is the missile *seeker's* notch gate, which
    does not inherit the launching aircraft's array. And a twenty-line
    scripted timeline beats every trained policy on every axis (0.96 vs 0.61 on
    a straight bandit, verified at 400 engagements on two held-out seeds), so
    the scripted pilot is what ships. Two real bugs were found on the way — the
    observation never told the policy *where* the incoming missile was, and
    checkpoint selection ran against an opponent nobody scores against, leaving
    survival as the whole signal — and fixing both moved the straight column
    0.07 → 0.65 without closing the gap. Tripling the training budget after
    that made it worse.
- **Energy action set** (`--action-set energy`): the 9 legacy maneuvers crossed
  with a throttle axis (burner / hold / idle) for 27 actions, with speed coupled
  to the flight path — `dv/dt = throttle*4.0 - g*sin(gamma) - 3.0*(turn/10 deg)`,
  clamped to 120–400 m/s. The legacy set holds speed fixed, so every turn is
  free and a fight between equals cannot resolve; this is the trade that makes
  an energy fight an energy fight. `QNetwork` is now shaped by its constructor
  and `load` reads the layer sizes back out of the checkpoint, so `energy`
  networks (216→27) and `legacy` networks (72→9) coexist; `--init` refuses a
  mismatched checkpoint and `run_pilot` detects the action set from the file and
  drives the throttle loop from the policy's choice. `legacy` stays the default
  and is unchanged.
- **`--transfer-init`**: lift a trained legacy 72→9 policy into the 216→27
  energy network (each input block copied into the maneuver's three throttle
  slots at 1/3 weight, each output column replicated three times), so an energy
  run starts from the legacy policy's maneuver preferences with only the
  throttle axis left to learn. Agrees with the source policy's maneuver choice
  ~97% of the time.
- **Self-play opponent** (`--opponent selfplay`): the bandit is flown by a
  *frozen* policy checkpoint instead of a scripted intent, so the opponent is
  exactly as capable as the agent. `--selfplay-init` picks the checkpoint
  (default: a frozen copy of the learner's starting weights),
  `--selfplay-refresh N` promotes the learner to be its own opponent every N
  episodes, and `--mixed-weights "selfplay=3,..."` folds it into the rehearsal
  curriculum. A plain `--opponent mixed` never draws self-play, so existing
  training commands behave exactly as before.
- **`--select-episodes`**: sample size for the periodic greedy eval that picks
  which weights get saved. It was hardcoded to 12, which is 2-3 samples per
  behavior once the curriculum draws five — the selection was mostly noise.
  Default unchanged at 12.
- **`ace` adaptive opponent**: a bandit that presses the attack while it
  holds the angular advantage and breaks away when it loses it. Added as an
  attempt at a harder benchmark; measured at **0.99 / 1.00** against the
  shipped policy, which had never trained on it, so it is *not* harder than
  the pure turning fight (0.95) — recorded here because the negative result
  is the useful part. `mixed` now includes `ace` in its draw, so `mixed`
  scores are not comparable across that change.
- **Reactive training opponents**: the training bandit can now `pursuit`
  (turn to point at the agent), go `evasive` (break toward the beam when
  threatened), or `mixed` (randomized per episode), via `--opponent` /
  `--eval-opponent`. The default stays straight-flying, so existing behavior
  is unchanged. This lets the policy learn to fight a maneuvering target
  instead of overfitting a straight one.
- **`--init` warm-start**: fine-tune training from an existing checkpoint
  instead of random weights (resume, or adapt a policy to new opponents).
- **Support diagnostics bundle** (`ucav-support`): collects version,
  OS/Python, license *status*, DCS addon install state, checkpoint
  fingerprint and the log tail into a zip for support requests. Secrets are
  never collected — the license key and API keys are reported as
  present/absent booleans only, and the bundle is plain text so users can
  inspect it before sending.
- **`--mixed-weights` rehearsal curriculum**: bias the per-episode behavior
  draw of `--opponent mixed` (e.g. `pursuit=3,straight=1,evasive=1`) to train
  one behavior hard while rehearsing the others, preventing catastrophic
  forgetting during fine-tuning.

### Added (continued)
- **Trained energy policy** (`checkpoints/ucav_policy_energy.npz`), rebuilt
  under the corrected climb physics: `--transfer-init` from the legacy policy,
  then the rehearsal curriculum with self-play weighted in. 400 engagements per
  behavior on a held-out seed — straight 0.93, pursuit 0.94, evasive 0.94, ace
  0.96, mixed 0.93, **self-play 0.81 (322 W / 67 L)** against the previous
  checkpoint's 0.35 (142 W / 236 L). This costs about four points on the
  scripted axes and is a deliberate exception to this project's
  no-regressions rule: the previous checkpoint was trained before the
  zoom-climb exploit was found, so its scores were earned in a world the
  simulator no longer has. Two further stages were run to buy those points
  back and both came out worse overall.
- **Speed-dependent climb authority** (`geometry.max_climb_angle`, energy action
  set only): the maximum climb angle scales from the full 70° at corner speed
  (200 m/s) down to level flight at `V_MIN`. Descending is never limited.

### Fixed
- **A hard speed floor made a vertical zoom climb free**, and it dominated
  self-play. Every unresolved mirror engagement — 60% of them — ended at the
  11 km ceiling, never at the horizontal boundary or the ground, with the
  bandit departing five times more often than the agent. Widening the arena to
  2,000 km and quadrupling the episode cap changed the outcome distribution by
  zero episodes, which ruled out the arena explanation this file previously
  offered. Climb authority is now bounded by the energy on hand. Running the
  shipped energy policy unchanged under the corrected physics moves boundary
  departures from 60% to ~1% and resolved fights from 40% to 60%; the rest are
  genuine stalemates.
- **The agent was penalized for the bandit's departure.** `out_of_bounds` cost
  −5 whenever *either* aircraft left the arena, so in a self-play fight the
  agent absorbed the penalty for something it does not control five times out
  of six. Departures are now attributed: `out_of_bounds` (−5) for the agent,
  the new `bandit_departed` (0.0) when only the bandit leaves. Applies to both
  action sets. Evaluation metrics are unaffected — neither outcome counts as a
  win or a conversion — so the published tables stand.
- **`delta_v2` was effectively unnormalized** — the legacy scaling divides it by
  1.0 while every other feature gets a real scale. Harmless in the legacy action
  set, where both aircraft are pinned at their start speed and the feature is
  identically zero, but with a throttle axis it reaches 145,000 against
  order-1 neighbors and swamps the first layer. The `energy` action set scales
  it like `v2`; the legacy constants are deliberately left untouched so old
  checkpoints stay valid.

### Changed
- **Upgraded the shipped policy** (`checkpoints/ucav_policy.npz`) again, and
  added a second checkpoint. Greedy eval over 400 engagements per behavior on
  each of **three** held-out seeds (mean shown), self-play scored against a
  frozen copy of the previous shipped policy:

  | Bandit | previous | `ucav_policy.npz` | `ucav_policy_robust.npz` |
  |---|---|---|---|
  | straight | 0.99 | **0.99** | 0.95 |
  | pursuit | 0.94 | **0.98** | **0.99** |
  | evasive | 1.00 | **1.00** | 0.96 |
  | ace | 0.98 | **1.00** | 0.99 |
  | mixed | 0.98 | **0.99** | 0.97 |
  | self-play | 0.01 | **0.32** | **0.70** |

  The default is a strict improvement — no scripted axis regresses (only the
  straight-bandit *conversion* rate moves, 1.00 → 0.99). `ucav_policy_robust`
  is the self-play-hardened alternative, selectable with `--checkpoint`: it
  trades ~4 points against straight and evasive targets for more than double
  the win rate against an opponent as capable as itself.
- **Corrected an over-claim about the previous policy.** Its 0.94–1.00 scores
  against scripted bandits did not mean it was strong: a policy trained
  specifically against it reached 0.89, and it scored 0.01 against a frozen
  copy of itself, with 92% of those engagements ending in `out_of_bounds`
  rather than a resolution. The scripted-opponent scores largely measured how
  predictable the opponents were.
- Earlier docs claimed the pursuit axis had been lifted to 0.95 and that the
  equal-speed pure-pursuit fight was not inherently unwinnable. The second
  claim stands (it is now 0.98–0.99); the first is superseded by the table
  above.
- **Config file**: `--config ucav_pilot.toml` supplies options from a TOML
  file (keys mirror the CLI flags); command-line flags still override it.
  Ships `ucav_pilot.example.toml`. Uses stdlib `tomllib` on Python 3.11+ and
  bundles the `tomli` backport automatically on 3.9/3.10.
- **Production logging**: lifecycle/error messages go to the console and a
  rotating file at `~/.ucav_pilot/logs/ucav_pilot.log`, controlled by
  `--log-level` and `--log-file`.
- **Packaging**: `pyproject.toml` makes the project pip-installable with
  console commands (`ucav-pilot`, `ucav-install`, `ucav-train`,
  `ucav-license`), optional-dependency extras (`radio`, `voice`, `plots`,
  `all`), a single-sourced version, and a `py.typed` marker. `MANIFEST.in`
  bundles the Lua addon and trained checkpoint into the source archive.
- **CI/CD**: GitHub Actions run the test suite on a Python 3.9–3.12 matrix and
  build the sdist/wheel on every push and PR; a tagged-release workflow
  publishes the distributions and an end-user install bundle.
- **One-click installer** (`install.bat` / `python -m dcs_bridge.install`):
  copies the Lua addon into the DCS `Saved Games\Scripts` folder and registers
  it in `Export.lua` idempotently, chaining any existing exports. Supports
  `--dry-run`, `--uninstall`, and an explicit `--saved-games` path.
- **Commercial licensing**: proprietary EULA (`LICENSE`) and a license-key
  scaffold (`dcs_bridge/licensing.py`). Offline-verifiable signed keys gate the
  paid **Pro** tier (LLM radio wingman + LLM WSO); without a key the product
  runs in **Trial** and falls back to the offline rule-based tier instead of
  refusing to start. New `--license-key` flag and `--version` flag.
- Editions/licensing documentation in the README.

## [0.1.0]

### Added
- DCS World addon connecting the RL air-combat agent as an in-sim AI pilot
  (Lua export ↔ Python UDP bridge at 20 Hz, target-trajectory prediction).
- LLM radio wingman (Claude) with Korean/English voice commands and an offline
  brevity-code fallback.
- CCA loyal-wingman teaming (MUM-T): named formations, flight-lead datalink,
  and leash-gated supervised-autonomy commit/rejoin.
- WSO back-seat text advisory (rule-based + optional LLM).
- Trained policy checkpoint, 105 unit tests, and setup/validation docs.
