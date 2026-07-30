# Changelog

All notable changes to UCAV AI Pilot are documented here. This project aims
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
