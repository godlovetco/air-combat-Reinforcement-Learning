"""Support diagnostics bundle for the UCAV AI Pilot.

When a customer reports a problem, ask them for a bundle::

    ucav-support                      # writes ucav-support-<timestamp>.zip
    ucav-support --out mybundle.zip
    ucav-support --print              # just show the diagnostics, write nothing

The bundle holds what is needed to reproduce a support case -- product
version, Python/OS, license *status*, whether the DCS addon is installed,
the policy checkpoint fingerprint, and the tail of the rotating log.

Secrets are never collected.  The license key itself is redacted to its
edition/licensee/expiry summary, and API keys are reported only as
"present / not set" booleans -- never their values.  Everything written is
plain text so the user can inspect the bundle before sending it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone
from typing import Optional

from . import __version__, licensing
from .install import default_saved_games_dirs
from .logging_setup import DEFAULT_LOG_FILE

# Environment variables whose *presence* is useful for support, but whose
# values are secrets and must never leave the user's machine.
_SECRET_ENV = ("ANTHROPIC_API_KEY", "UCAV_LICENSE_KEY", "UCAV_PRODUCT_SECRET")

LOG_TAIL_LINES = 400


def _file_fingerprint(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return {
        "path": path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def _license_summary() -> dict:
    """License *status* only -- the key itself is never included."""
    lic = licensing.load_license()
    return {
        "edition": lic.edition,
        "licensee": lic.licensee or "(unlicensed)",
        "features": list(lic.features),
        "expires": lic.expires,
        "valid": lic.valid,
        "is_trial": lic.is_trial,
        "note": lic.reason or "",
    }


def _dcs_addon_status() -> list:
    """Whether the Lua addon is installed in each detected DCS folder."""
    out = []
    for saved_games in default_saved_games_dirs():
        scripts = saved_games / "Scripts"
        export = scripts / "Export.lua"
        registered = False
        if export.is_file():
            try:
                from .install import BEGIN_MARK

                registered = BEGIN_MARK in export.read_text(
                    encoding="utf-8", errors="replace")
            except OSError:
                registered = False
        out.append({
            "saved_games": str(saved_games),
            "addon_script_installed":
                (scripts / "UCAVPilot" / "UCAVPilotExport.lua").is_file(),
            "export_lua_exists": export.is_file(),
            "loader_registered": registered,
        })
    return out


def _log_tail(log_file: str, lines: int = LOG_TAIL_LINES) -> str:
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError as exc:
        return f"(no log available at {log_file}: {exc})"


def collect_diagnostics(checkpoint: str = "checkpoints/ucav_policy.npz") -> dict:
    """Gather support diagnostics. Contains no secrets (see module docstring)."""
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": {"name": "UCAV AI Pilot", "version": __version__},
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "license": _license_summary(),
        "checkpoint": _file_fingerprint(checkpoint) or {"path": checkpoint,
                                                        "missing": True},
        "dcs": _dcs_addon_status(),
        # Presence only -- values are secrets and are deliberately excluded.
        "env_present": {name: (name in os.environ) for name in _SECRET_ENV},
        "optional_packages": _optional_packages(),
    }


def _optional_packages() -> dict:
    found = {}
    for mod in ("numpy", "anthropic", "matplotlib", "speech_recognition", "pyttsx3"):
        try:
            __import__(mod)
            found[mod] = True
        except Exception:
            found[mod] = False
    return found


def write_bundle(out_path: str, checkpoint: str = "checkpoints/ucav_policy.npz",
                 log_file: str = DEFAULT_LOG_FILE) -> str:
    """Write a zip bundle and return its path."""
    diagnostics = collect_diagnostics(checkpoint=checkpoint)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("diagnostics.json", json.dumps(diagnostics, indent=2))
        zf.writestr("ucav_pilot.log.tail", _log_tail(log_file))
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, metavar="ZIP",
                   help="bundle path (default: ucav-support-<timestamp>.zip)")
    p.add_argument("--checkpoint", default="checkpoints/ucav_policy.npz")
    p.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="print diagnostics to stdout instead of writing a zip")
    args = p.parse_args(argv)

    if args.print_only:
        print(json.dumps(collect_diagnostics(checkpoint=args.checkpoint), indent=2))
        return 0

    out = args.out or datetime.now(timezone.utc).strftime(
        "ucav-support-%Y%m%d-%H%M%S.zip")
    write_bundle(out, checkpoint=args.checkpoint, log_file=args.log_file)
    print(f"support bundle written to {out}")
    print("It contains no license key or API key -- you can inspect it before sending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
