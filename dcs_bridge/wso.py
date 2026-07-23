"""Back-seat WSO (Weapon Systems Officer) text advisory.

In WSO mode the *human* flies the jet and the AI rides in back, calling the
fight over the intercom as text: threat warnings, BRA (bearing/range/
altitude) calls, weapons-employment cues, energy/geometry advice, and — its
signature contribution — the trained RL policy's *recommended maneuver*
rendered in plain language ("recommend come right to 210, nose up").  It is
purely advisory: it never touches the controls.

Two advisors are provided, mirroring the radio wingman:

* :class:`WSOAdvisor` -- a rule-based back-seater with prioritized, cooldown-
  throttled calls in Korean or English.  No dependencies; always available
  and fast enough to run inside the 20 Hz loop, so time-critical calls
  ("break!", "guns") are never late.
* :class:`WSOAgent` -- an optional Claude-backed advisor for free-form,
  situation-aware commentary; used on a background timer so its latency
  never stalls the flight loop.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

# Employment envelopes (shared with the flight loop's gun logic).
GUN_RANGE = 1_200.0     # m
GUN_ASPECT = 4.0        # deg, own aspect (nose-on the bandit)
IN_RANGE_MIN = 1_500.0  # m, near edge of a notional missile band
IN_RANGE_MAX = 8_000.0  # m, far edge
SHOT_ASPECT = 35.0      # deg, own aspect for a valid forward-quarter shot

# Energy / safety thresholds.
CORNER_MS = 180.0       # ~ best-turn speed for a generic fighter
LOW_ENERGY_MS = 150.0   # below this in a hard turn -> bleeding out
LOW_ALT_M = 300.0       # AGL-ish floor for a "altitude!" call
THREAT_RANGE = 5_000.0  # m, bandit nose-on inside this = spike
MERGE_RANGE = 2_500.0   # m, break-now range


def _turn_word(heading_error_deg: float, lang: str) -> str:
    """'right'/'left' (or Korean) for a signed heading error to the target."""
    right = heading_error_deg >= 0.0
    if lang == "ko":
        return "우로" if right else "좌로"
    return "right" if right else "left"


def _pitch_word(gamma_deg: float, lang: str) -> str:
    if gamma_deg > 5.0:
        return "기수 올려" if lang == "ko" else "nose up"
    if gamma_deg < -5.0:
        return "기수 내려" if lang == "ko" else "nose down"
    return "수평 유지" if lang == "ko" else "level"


def _heading_error(target_deg: float, current_deg: float) -> float:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


class WSOAdvisor:
    """Rule-based back-seater: turns the tactical picture into one text call.

    Call :meth:`advise` every tick (or every ``period`` seconds) with a
    situation dict; it returns the single most important new call, or None
    when everything on the board is still on cooldown.  Calls are prioritized
    (lower number = more urgent) and each key is rate-limited so the WSO does
    not chatter.
    """

    # key -> cooldown seconds
    _COOLDOWN = {
        "altitude": 3.0,
        "break": 3.0,
        "spike": 4.0,
        "guns": 2.0,
        "in_range": 6.0,
        "energy": 8.0,
        "overshoot": 6.0,
        "maneuver": 4.0,
        "bra": 10.0,
        "station": 12.0,
    }

    def __init__(self, lang: str = "ko"):
        self.lang = "ko" if str(lang).lower().startswith("ko") else "en"
        self._last: Dict[str, float] = {}
        self._had_bandit = False
        self._prev_range: Optional[Tuple[str, float, float]] = None  # name, t, range

    # ------------------------------------------------------------------ #
    def advise(self, sit: dict) -> Optional[str]:
        t = float(sit.get("t", 0.0))
        bandit = sit.get("bandit")
        closure = self._closure(t, bandit)

        candidates = self._collect(t, sit, closure)

        # Contact / no-joy are edge-triggered: they fire once when the picture
        # changes and bypass cooldown, but still yield to a "break!" or
        # "altitude!" that outranks them.
        if bandit is not None and not self._had_bandit:
            candidates.append((3.5, "bra", self._contact_call(bandit), True))
        elif bandit is None and self._had_bandit:
            candidates.append((3.5, "bra", self._no_joy(), True))
        self._had_bandit = bandit is not None

        candidates.sort(key=lambda c: c[0])  # by priority
        for _prio, key, text, force in candidates:
            if force or self._ready(key, t):
                return self._say(key, t, text)
        return None

    # ------------------------------------------------------------------ #
    def _collect(self, t: float, sit: dict, closure: float
                 ) -> List[Tuple[float, str, str, bool]]:
        out: List[Tuple[float, str, str, bool]] = []
        own = sit.get("own", {})
        alt = float(own.get("altitude_m", 9999.0))
        tas = float(own.get("tas", 250.0))
        vv = float(own.get("vv", 0.0))
        bank = abs(float(own.get("bank_deg", 0.0)))
        bandit = sit.get("bandit")

        ko = self.lang == "ko"

        # 1 - ground / altitude safety
        if alt < LOW_ALT_M and vv < 0.0:
            out.append((1, "altitude",
                        "고도! 고도! 기수 당겨!" if ko else "Altitude! Altitude, pull up!",
                        False))

        # 2 - defensive: bandit nose-on and closing
        if bandit is not None:
            rng = float(bandit["range_m"])
            b_aspect = float(bandit.get("bandit_aspect_deg", 180.0))
            if b_aspect < 30.0 and rng < THREAT_RANGE and closure > 0.0:
                if rng < MERGE_RANGE:
                    out.append((2, "break",
                                ("브레이크! 지금 돌려, 채프 플레어!" if ko
                                 else "Break now, chaff and flares!"), False))
                else:
                    out.append((2, "spike",
                                (f"스파이크, {rng/1000:.0f}킬로. 브레이크 준비." if ko
                                 else f"Spike, {rng/1000:.0f} k, get ready to break."), False))

        # 3 - weapons employment
        if bandit is not None:
            rng = float(bandit["range_m"])
            o_aspect = float(bandit.get("own_aspect_deg", 180.0))
            wf = bool(sit.get("weapons_free", False))
            if rng < GUN_RANGE and o_aspect < GUN_ASPECT:
                out.append((3, "guns",
                            ("건 사정권, 파라미터 안. 사격!" if ko
                             else "In gun parameters — guns, guns!") if wf else
                            ("건 사정권. 사격 허가 요청." if ko
                             else "In gun parameters — request weapons free."), False))
            elif IN_RANGE_MIN < rng < IN_RANGE_MAX and o_aspect < SHOT_ASPECT:
                out.append((3, "in_range",
                            (f"사격 범위, {rng/1000:.0f}킬로, 조준 유지." if ko
                             else f"In range, {rng/1000:.0f} k, keep the pipper on."),
                            False))

        # 4 - energy / corner
        if bank > 60.0:
            if tas < LOW_ENERGY_MS:
                out.append((4, "energy",
                            ("에너지 부족, 기수 내려 속도 회복." if ko
                             else "Low and slow, unload to regain energy."), False))
            elif tas > CORNER_MS + 60.0:
                out.append((4, "energy",
                            ("코너 속도 초과, 당겨서 선회율 높여." if ko
                             else "Above corner, pull for more rate."), False))

        # 5 - geometry: overshoot
        if bandit is not None:
            rng = float(bandit["range_m"])
            o_aspect = float(bandit.get("own_aspect_deg", 180.0))
            if o_aspect > 60.0 and rng < 1_500.0 and closure > 30.0:
                out.append((5, "overshoot",
                            ("오버슈트 주의, 하이 요요로 에너지 살려." if ko
                             else "Overshooting — high yo-yo, preserve turn room."), False))

        # 6 - the RL policy's recommended maneuver
        rec = sit.get("recommend")
        if rec is not None:
            he = _heading_error(float(rec["heading_deg"]), float(own.get("heading_deg", 0.0)))
            if abs(he) > 8.0 or abs(float(rec.get("gamma_deg", 0.0))) > 8.0:
                turn = _turn_word(he, self.lang)
                pitch = _pitch_word(float(rec.get("gamma_deg", 0.0)), self.lang)
                tgt = int(round(float(rec["heading_deg"]))) % 360
                out.append((6, "maneuver",
                            (f"추천: {turn} {tgt:03d}, {pitch}." if ko
                             else f"Recommend come {turn} to {tgt:03d}, {pitch}."), False))

        # 7 - periodic BRA
        if bandit is not None:
            out.append((7, "bra", self._bra(bandit), False))

        # 8 - formation station keeping (no bandit)
        lead = sit.get("lead")
        if bandit is None and lead is not None:
            err = float(lead.get("station_error_m", 0.0))
            if err > 500.0:
                out.append((8, "station",
                            ("편대 위치 이탈, 대형 좁혀." if ko
                             else "Off station, tighten the formation."), False))
        return out

    # ------------------------------------------------------------------ #
    def _bra(self, bandit: dict) -> str:
        brg = int(round(float(bandit.get("bearing_deg", 0.0)))) % 360
        rng = float(bandit["range_m"]) / 1000.0
        alt = int(round(float(bandit.get("altitude_m", 0.0))))
        if self.lang == "ko":
            return f"브라 {brg:03d}, {rng:.0f}킬로, 고도 {alt}미터."
        return f"BRA {brg:03d}, {rng:.0f} k, {alt} meters."

    def _contact_call(self, bandit: dict) -> str:
        name = bandit.get("name", "bandit")
        head = "컨택" if self.lang == "ko" else "Contact"
        return f"{head}, {name}. " + self._bra(bandit)

    def _no_joy(self) -> str:
        return "노 조이, 컨택 상실." if self.lang == "ko" else "No joy, lost contact."

    def _closure(self, t: float, bandit: Optional[dict]) -> float:
        """Closure rate (m/s, positive = closing) from successive ranges."""
        if bandit is None:
            self._prev_range = None
            return 0.0
        name = bandit.get("name", "?")
        rng = float(bandit["range_m"])
        prev = self._prev_range
        self._prev_range = (name, t, rng)
        if prev is None or prev[0] != name or t <= prev[1]:
            return 0.0
        return (prev[2] - rng) / (t - prev[1])

    def _ready(self, key: str, t: float) -> bool:
        last = self._last.get(key)
        return last is None or (t - last) >= self._COOLDOWN.get(key, 6.0)

    def _say(self, key: str, t: float, text: str) -> str:
        self._last[key] = t
        return text


# ---------------------------------------------------------------------- #
# Optional LLM back-seater
# ---------------------------------------------------------------------- #
DEFAULT_MODEL = "claude-opus-4-8"

WSO_SYSTEM_PROMPT = """\
You are the WSO (Weapon Systems Officer) in the back seat of a fighter. A
human pilot is flying; you advise them over the intercom. Each message gives
you a <situation> block with the live tactical picture (own state, bandit
BRA/aspect/range, weapons state, and the RL autopilot's recommended maneuver).

Give ONE short back-seat call, like real WSO comms: a threat warning, a BRA
call, a weapons cue, an energy/geometry tip, or the recommended maneuver in
plain words ("come right to 210, nose up"). Rules:
- Under 15 words. This is intercom, not a lecture.
- Reply in the pilot's language (Korean or English) as set by the situation.
- Only state facts present in the situation block; never invent contacts.
- If nothing is new or worth saying, reply with an empty message.
- You are advisory only. You do not fly the jet.
"""


class WSOAgent:
    """Claude-backed back-seater for free-form advisory commentary."""

    def __init__(self, model: str = DEFAULT_MODEL, client=None, lang: str = "ko"):
        if client is None:
            import anthropic  # deferred; offline mode needs no install

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.lang = "ko" if str(lang).lower().startswith("ko") else "en"

    def advise(self, sit: dict) -> Optional[str]:
        payload = dict(sit)
        payload["language"] = "Korean" if self.lang == "ko" else "English"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=WSO_SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": f"<situation>{json.dumps(payload, ensure_ascii=False)}</situation>",
            }],
        )
        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text += block.text
        text = text.strip()
        return text or None


def make_wso(lang: str = "ko", model: str = DEFAULT_MODEL, use_llm: bool = False):
    """WSOAgent when an LLM is requested and reachable, else WSOAdvisor."""
    if use_llm:
        try:
            return WSOAgent(model=model, lang=lang)
        except Exception as exc:  # no SDK / no credentials
            print(f"WSO: Claude unavailable ({exc}); using rule-based back-seater")
    return WSOAdvisor(lang=lang)
