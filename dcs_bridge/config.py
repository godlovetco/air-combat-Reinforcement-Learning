"""TOML configuration file support for the UCAV AI Pilot.

Non-CLI users (and repeatable setups) can put their options in a
``ucav_pilot.toml`` file instead of typing flags every time::

    # ucav_pilot.toml
    checkpoint   = "checkpoints/ucav_policy.npz"
    weapons      = true
    formation    = "combat_spread"
    leash        = "tight"
    radio        = true
    license_key  = "UCAV1.xxxx.yyyy"
    log_level    = "INFO"

Keys are the long CLI option names with dashes turned into underscores
(``--radio-lang`` -> ``radio_lang``).  Explicit command-line flags always win
over the file; the file wins over built-in defaults.

Parsing uses the standard-library ``tomllib`` (Python 3.11+), falling back to
the ``tomli`` backport when installed (``pip install "ucav-ai-pilot[config]"``
on 3.9/3.10).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class ConfigError(Exception):
    """Raised for a missing/invalid config file or unknown keys."""


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib as toml  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as toml  # backport for 3.9 / 3.10
        except ModuleNotFoundError as exc:
            raise ConfigError(
                "reading a TOML config needs Python 3.11+ or the 'tomli' "
                "package (pip install tomli)"
            ) from exc
    try:
        with open(path, "rb") as fh:
            return toml.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except Exception as exc:  # tomllib.TOMLDecodeError and friends
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def load_config(path: str, valid_keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a config file into a flat ``{dest: value}`` dict.

    Accepts either flat top-level keys or keys grouped under any ``[section]``
    tables (the section names are ignored -- they are just for readability).
    Dashes in keys are normalized to underscores.  If ``valid_keys`` is given,
    any key not in it raises :class:`ConfigError` (catches typos early).
    """
    raw = _load_toml(path)
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):  # a [section] table -> merge its members
            for subkey, subvalue in value.items():
                flat[subkey.replace("-", "_")] = subvalue
        else:
            flat[key.replace("-", "_")] = value

    if valid_keys is not None:
        allowed = set(valid_keys)
        unknown = sorted(k for k in flat if k not in allowed)
        if unknown:
            raise ConfigError(
                f"unknown config option(s): {', '.join(unknown)} in {path}"
            )
    return flat
