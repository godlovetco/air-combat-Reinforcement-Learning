# Changelog

All notable changes to UCAV AI Pilot are documented here. This project aims
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
