"""Product licensing for the UCAV AI Pilot (commercial edition).

The reinforcement-learning pilot, the offline rule-based radio and the
rule-based WSO back-seater all run without a license (TRIAL).  The paid
**Pro** tier unlocks the Claude-powered features -- the LLM radio wingman and
the LLM WSO advisor -- gated by the ``llm`` feature on a valid license key.
Without a key the product degrades gracefully to its offline tier rather than
refusing to run, so a trial is always usable.

License keys
------------
A key is a compact, offline-verifiable token::

    UCAV1.<base64url(payload)>.<base64url(hmac-sha256 tag)>

``payload`` is a small JSON object ``{ed, lic, feat, exp}`` (edition,
licensee, features, expiry).  The vendor issues keys with
:func:`issue_key`; the shipped client verifies them with :func:`verify_key`.

Security model (scaffold)
-------------------------
Signing uses HMAC-SHA256 with a product secret read from
``UCAV_PRODUCT_SECRET``.  Keep that secret only on the vendor's key-issuing
machine/server; the placeholder default here is for development and MUST be
replaced for a real release.  Because HMAC is symmetric, a shipped client
that can verify can in principle also forge -- for stronger offline
protection swap the HMAC for an Ed25519 signature (public key in the client,
private key held by the vendor) or verify keys against a licensing server.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple

# Feature flags a license may grant.
FEATURE_CORE = "core"   # RL pilot + offline rule-based radio/WSO (always on)
FEATURE_CCA = "cca"     # loyal-wingman formation teaming
FEATURE_WSO = "wso"     # rule-based back-seat advisory
FEATURE_LLM = "llm"     # Claude radio wingman + LLM WSO  (the paid tier)
FEATURES = (FEATURE_CORE, FEATURE_CCA, FEATURE_WSO, FEATURE_LLM)

EDITIONS = ("trial", "standard", "pro")
KEY_PREFIX = "UCAV1"

_DEV_SECRET = "UCAV-DEV-PLACEHOLDER-SECRET-CHANGE-ME"
_KEY_FILE = os.path.join(os.path.expanduser("~"), ".ucav_pilot", "license.key")


def _product_secret(secret: Optional[bytes] = None) -> bytes:
    if secret is not None:
        return secret
    return os.environ.get("UCAV_PRODUCT_SECRET", _DEV_SECRET).encode("utf-8")


class LicenseError(Exception):
    """Raised for a malformed or tampered license key."""


@dataclass
class License:
    edition: str
    licensee: str
    features: Tuple[str, ...]
    expires: Optional[str]        # ISO date "YYYY-MM-DD", or None = perpetual
    valid: bool = True
    reason: str = ""              # why it is invalid / running as trial

    def has(self, feature: str) -> bool:
        return self.valid and feature in self.features

    @property
    def is_trial(self) -> bool:
        return self.edition == "trial"

    def describe(self) -> str:
        who = self.licensee or "unlicensed"
        exp = f", expires {self.expires}" if self.expires else ""
        feats = ", ".join(self.features) if self.features else "none"
        tail = f"  [{self.reason}]" if self.reason else ""
        return f"{self.edition} edition ({who}{exp}); features: {feats}{tail}"


# Trial: everything except the paid LLM tier.
TRIAL = License(
    edition="trial",
    licensee="",
    features=(FEATURE_CORE, FEATURE_CCA, FEATURE_WSO),
    expires=None,
    valid=True,
    reason="no license key -- Pro (LLM) features disabled",
)


# ---------------------------------------------------------------------- #
def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _sign(payload_b64: str, secret: Optional[bytes] = None) -> str:
    tag = hmac.new(_product_secret(secret), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64u_encode(tag[:16])


def issue_key(
    edition: str,
    licensee: str,
    features: Sequence[str],
    expires: Optional[str] = None,
    secret: Optional[bytes] = None,
) -> str:
    """Vendor-side: mint a signed license key. Requires the product secret."""
    if edition not in EDITIONS:
        raise ValueError(f"unknown edition {edition!r}, expected {EDITIONS}")
    for feat in features:
        if feat not in FEATURES:
            raise ValueError(f"unknown feature {feat!r}, expected {FEATURES}")
    payload = {"ed": edition, "lic": licensee, "feat": list(features), "exp": expires}
    payload_b64 = _b64u_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return f"{KEY_PREFIX}.{payload_b64}.{_sign(payload_b64, secret)}"


def verify_key(key: str, now: Optional[datetime] = None, secret: Optional[bytes] = None) -> License:
    """Client-side: verify a key's signature and expiry.

    Returns a :class:`License` (``valid=False`` if merely expired).  Raises
    :class:`LicenseError` for a malformed or tampered key.
    """
    parts = key.strip().split(".")
    if len(parts) != 3:
        raise LicenseError("malformed license key")
    prefix, payload_b64, sig = parts
    if prefix != KEY_PREFIX:
        raise LicenseError(f"unknown license key version {prefix!r}")
    if not hmac.compare_digest(_sign(payload_b64, secret), sig):
        raise LicenseError("invalid license signature")
    try:
        payload = json.loads(_b64u_decode(payload_b64))
    except Exception as exc:  # noqa: BLE001 - any decode failure is corruption
        raise LicenseError(f"corrupt license payload ({exc})")

    edition = payload.get("ed", "standard")
    licensee = payload.get("lic", "")
    features = tuple(payload.get("feat", []))
    expires = payload.get("exp")
    if expires:
        today = (now or datetime.now(timezone.utc)).date().isoformat()
        if today > expires:
            return License(edition, licensee, features, expires,
                           valid=False, reason=f"license expired {expires}")
    return License(edition, licensee, features, expires, valid=True)


def _read_key_file(path: str = _KEY_FILE) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def load_license(
    key: Optional[str] = None,
    now: Optional[datetime] = None,
    key_file: str = _KEY_FILE,
) -> License:
    """Resolve the active license from (in order): explicit key, the
    ``UCAV_LICENSE_KEY`` env var, the key file, else TRIAL.

    Never raises: an invalid/expired key logs a note and falls back to TRIAL
    so the product always starts.
    """
    key = key or os.environ.get("UCAV_LICENSE_KEY") or _read_key_file(key_file)
    if not key:
        return TRIAL
    try:
        lic = verify_key(key, now=now)
    except LicenseError as exc:
        print(f"license: {exc}; running in trial mode")
        return TRIAL
    if not lic.valid:
        print(f"license: {lic.reason}; running in trial mode")
        return TRIAL
    return lic


# ---------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="UCAV AI Pilot license tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    iss = sub.add_parser("issue", help="vendor: mint a license key")
    iss.add_argument("--edition", default="pro", choices=EDITIONS)
    iss.add_argument("--licensee", required=True, help="customer name / order id")
    iss.add_argument("--features", default="core,cca,wso,llm",
                     help="comma-separated feature list")
    iss.add_argument("--expires", default=None, help="YYYY-MM-DD (omit = perpetual)")

    chk = sub.add_parser("verify", help="check a license key")
    chk.add_argument("key")

    args = p.parse_args()
    if args.cmd == "issue":
        feats = [f.strip() for f in args.features.split(",") if f.strip()]
        print(issue_key(args.edition, args.licensee, feats, args.expires))
    elif args.cmd == "verify":
        try:
            lic = verify_key(args.key)
        except LicenseError as exc:
            print(f"INVALID: {exc}")
            raise SystemExit(1)
        print(("VALID  " if lic.valid else "EXPIRED ") + lic.describe())
        raise SystemExit(0 if lic.valid else 1)


if __name__ == "__main__":
    _main()
