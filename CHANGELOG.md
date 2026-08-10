# Changelog

All notable changes to UCAV AI Pilot are documented here. This project aims
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Reactive training opponents**: the training bandit can now `pursuit`
  (turn to point at the agent), go `evasive` (break toward the beam when
  threatened), or `mixed` (randomized per episode), via `--opponent` /
  `--eval-opponent`. The default stays straight-flying, so existing behavior
  is unchanged. This lets the policy learn to fight a maneuvering target
  instead of overfitting a straight one.
- **`--init` warm-start**: fine-tune training from an existing checkpoint
  instead of random weights (resume, or adapt a policy to new opponents).
- **`--mixed-weights` rehearsal curriculum**: bias the per-episode behavior
  draw of `--opponent mixed` (e.g. `pursuit=3,straight=1,evasive=1`) to train
  one behavior hard while rehearsing the others, preventing catastrophic
  forgetting during fine-tuning.

### Changed
- **Upgraded the shipped policy** (`checkpoints/ucav_policy.npz`), twice:
  first fine-tuned against `mixed` reactive opponents (straight 0.72→0.96
  win, pursuit 0.00→0.14), then through a pursuit-heavy rehearsal
  curriculum (`--mixed-weights "pursuit=3,straight=1,evasive=1"`). Final
  greedy eval over 240 randomized engagements per behavior: **0.96 / 0.97**
  win/conversion vs a straight bandit, **0.99 / 0.99** vs evasive,
  **0.81 / 0.81** vs mixed, and **0.35 / 0.36** vs pure pursuit — the
  hardest case, up from 0.00 for the original straight-only-trained policy,
  with the other profiles held.
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
