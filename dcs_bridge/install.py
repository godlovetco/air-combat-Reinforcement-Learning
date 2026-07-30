"""One-click installer for the UCAV AI Pilot DCS addon.

Copies the Lua export script into your DCS "Saved Games" Scripts folder and
registers it in ``Export.lua`` so DCS loads it on the next mission.  The
registration is idempotent and chains any exports you already run (Tacview,
SRS, DCS-BIOS, ...), so nothing you had stops working.

    python -m dcs_bridge.install                 # auto-detect DCS and install
    python -m dcs_bridge.install --dry-run       # show what would happen
    python -m dcs_bridge.install --uninstall     # remove the addon
    python -m dcs_bridge.install --saved-games "D:/Saved Games/DCS.openbeta"

On Windows, double-clicking ``install.bat`` runs this for you.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

BEGIN_MARK = "-- >>> UCAV AI Pilot loader (managed by installer) >>>"
END_MARK = "-- <<< UCAV AI Pilot loader (managed by installer) <<<"

# The loader body registered inside Export.lua.  Kept identical in spirit to
# dcs-addon/Export.lua, but self-contained between the managed markers so the
# installer can add/replace/remove it safely.
LOADER_BODY = """\
do
    local ucavPilotScript = lfs.writedir() .. [[Scripts\\UCAVPilot\\UCAVPilotExport.lua]]
    local f = io.open(ucavPilotScript, "r")
    if f then
        f:close()
        local ok, err = pcall(dofile, ucavPilotScript)
        if not ok and log and log.write then
            log.write("UCAVPilot", log.ERROR, "failed to load: " .. tostring(err))
        end
    end
end"""


def managed_block() -> str:
    return f"{BEGIN_MARK}\n{LOADER_BODY}\n{END_MARK}\n"


def default_addon_src() -> Path:
    """The repository's ``dcs-addon`` directory (bundled with this package)."""
    return Path(__file__).resolve().parent.parent / "dcs-addon"


def default_saved_games_dirs() -> List[Path]:
    """Candidate DCS 'Saved Games' directories on this machine (Windows)."""
    candidates: List[Path] = []
    userprofile = os.environ.get("USERPROFILE")
    bases = [Path(userprofile)] if userprofile else []
    # Fall back to the home dir (also covers a mounted Windows profile).
    bases.append(Path(os.path.expanduser("~")))
    for base in bases:
        for variant in ("DCS", "DCS.openbeta"):
            d = base / "Saved Games" / variant
            if d.is_dir() and d not in candidates:
                candidates.append(d)
    return candidates


def _strip_managed_block(text: str) -> str:
    """Remove a previously installed managed block (and its trailing blank)."""
    if BEGIN_MARK not in text or END_MARK not in text:
        return text
    start = text.index(BEGIN_MARK)
    end = text.index(END_MARK) + len(END_MARK)
    cleaned = text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")
    return cleaned.strip("\n") + "\n" if cleaned.strip() else ""


def register_export(export_lua: Path, dry_run: bool = False) -> str:
    """Ensure Export.lua contains exactly one managed loader block.

    Preserves any other content.  Returns "created", "updated", or
    "unchanged".
    """
    existing = export_lua.read_text(encoding="utf-8") if export_lua.exists() else ""
    had_block = BEGIN_MARK in existing
    base = _strip_managed_block(existing)
    block = managed_block()
    if base.strip():
        new_text = base.rstrip("\n") + "\n\n" + block
    else:
        new_text = block
    if new_text == existing:
        return "unchanged"
    if not dry_run:
        export_lua.parent.mkdir(parents=True, exist_ok=True)
        export_lua.write_text(new_text, encoding="utf-8")
    return "updated" if had_block else "created"


def install(
    saved_games: Path,
    addon_src: Optional[Path] = None,
    dry_run: bool = False,
) -> List[str]:
    """Install the addon into ``saved_games``. Returns a list of actions."""
    addon_src = Path(addon_src) if addon_src else default_addon_src()
    src_pkg = addon_src / "Scripts" / "UCAVPilot"
    if not (src_pkg / "UCAVPilotExport.lua").is_file():
        raise FileNotFoundError(f"addon source not found under {addon_src}")

    saved_games = Path(saved_games)
    dest_scripts = saved_games / "Scripts"
    dest_pkg = dest_scripts / "UCAVPilot"
    actions: List[str] = []

    actions.append(f"copy {src_pkg} -> {dest_pkg}")
    if not dry_run:
        dest_pkg.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_pkg, dest_pkg, dirs_exist_ok=True)

    outcome = register_export(dest_scripts / "Export.lua", dry_run=dry_run)
    actions.append(f"{outcome} loader in {dest_scripts / 'Export.lua'}")
    return actions


def uninstall(saved_games: Path, dry_run: bool = False) -> List[str]:
    """Remove the addon from ``saved_games``. Returns a list of actions."""
    saved_games = Path(saved_games)
    dest_scripts = saved_games / "Scripts"
    dest_pkg = dest_scripts / "UCAVPilot"
    export_lua = dest_scripts / "Export.lua"
    actions: List[str] = []

    if export_lua.exists():
        text = export_lua.read_text(encoding="utf-8")
        if BEGIN_MARK in text:
            actions.append(f"remove loader from {export_lua}")
            if not dry_run:
                cleaned = _strip_managed_block(text)
                if cleaned.strip():
                    export_lua.write_text(cleaned, encoding="utf-8")
                else:
                    export_lua.unlink()  # we created it and it's now empty
    if dest_pkg.is_dir():
        actions.append(f"delete {dest_pkg}")
        if not dry_run:
            shutil.rmtree(dest_pkg)
    if not actions:
        actions.append("nothing to remove")
    return actions


def _resolve_targets(args) -> List[Path]:
    if args.saved_games:
        return [Path(args.saved_games)]
    found = default_saved_games_dirs()
    if not found:
        raise SystemExit(
            "could not find a DCS 'Saved Games' folder automatically.\n"
            "Pass it explicitly, e.g.:\n"
            '  python -m dcs_bridge.install --saved-games '
            '"%USERPROFILE%/Saved Games/DCS"'
        )
    return found


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saved-games", default=None,
                   help="DCS Saved Games folder (auto-detected if omitted)")
    p.add_argument("--addon-src", default=None,
                   help="override the bundled dcs-addon source directory")
    p.add_argument("--uninstall", action="store_true", help="remove the addon")
    p.add_argument("--dry-run", action="store_true",
                   help="print actions without changing anything")
    args = p.parse_args(argv)

    targets = _resolve_targets(args)
    tag = "[dry-run] " if args.dry_run else ""
    for target in targets:
        print(f"{tag}{'Uninstalling from' if args.uninstall else 'Installing to'}: {target}")
        fn = uninstall if args.uninstall else install
        kwargs = {} if args.uninstall else {"addon_src": args.addon_src}
        for action in fn(target, dry_run=args.dry_run, **kwargs):
            print(f"  {tag}{action}")

    if not args.uninstall:
        print("\nDone. Start a DCS mission with a player aircraft, then run:")
        print("  python -m dcs_bridge.run_pilot --checkpoint checkpoints/ucav_policy.npz")
        print("Enter a Pro license key with --license-key to unlock the LLM wingman.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
