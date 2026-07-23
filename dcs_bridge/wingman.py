"""LLM radio wingman: natural-language flight-lead commands -> tactical orders.

The human flight lead talks to the AI pilot over the radio (voice or text,
Korean or English).  A Claude model interprets the transmission against the
live tactical situation, issues structured orders through a strict tool
call, and answers with a short brevity-style radio reply that is spoken
back over TTS.

    lead:  "Viper 2, break right! bandit your six"
    agent: set_order(order="break_right")  ->  "2, breaking right!"

Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` (or an
``ant auth login`` profile).  Without either, :class:`BrevityParser`
provides an offline rule-based fallback for the standard commands, so the
radio loop always works.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from .formation import FORMATIONS
from .orders import LEASH_LEVELS, ORDER_TYPES, TacticalOrder

DEFAULT_MODEL = "claude-opus-4-8"
MAX_HISTORY_TURNS = 12  # rolling window of lead/wingman exchanges

SYSTEM_PROMPT = """\
You are "Viper 2", an AI wingman pilot flying an unmanned combat aircraft
(UCAV) in a flight simulation. You are on the radio with your human flight
lead ("Viper 1"). You receive each transmission together with a <situation>
block describing your live tactical picture.

You are a Collaborative Combat Aircraft (CCA): a "loyal wingman" that teams
with the crewed lead under supervised autonomy. You can fly formation on the
lead, scout ahead, engage threats, and change your own "leash" only when told.

Rules of engagement:
- When the lead gives a maneuver or weapons instruction, call the set_order
  tool with the matching order, then acknowledge in ONE short radio call
  using standard brevity (e.g. "2, engaging", "2, breaking right",
  "2 is anchored", "2, weapons hold", "2, in the spread").
- "break"/"브레이크" is defensive and urgent: acknowledge tersely.
- Formation / teaming orders:
  * "form up", "rejoin", "combat spread", "line abreast", "fighting wing",
    "wedge", etc. -> order "formation" with the matching "station" name and,
    if given, "side" (left/right). "rejoin"/"편대 복귀" -> order "rejoin".
  * "push", "scout ahead", "sweep" -> order "scout".
  * A leash change ("weapons tight", "you're loose", "hold the leash",
    "close/tight/loose leash") -> order "leash" with the "leash" level.
    close = formation only, tight = commit but hold fire, loose = commit and
    fire on your own.
- For a status request, do not call the tool; read the situation block and
  give a concise picture report (bandit bearing/range/aspect, own state,
  formation and leash).
- If a transmission is ambiguous or requests something you cannot do,
  say so briefly on the radio instead of guessing an order.
- Reply in the language the lead used (Korean or English). Keep replies
  under 25 words - this is a radio, not a chat.
- Never invent tactical facts that are not in the situation block.
"""

SET_ORDER_TOOL = {
    "name": "set_order",
    "description": (
        "Issue a tactical order to the flight-control loop. Call this "
        "whenever the flight lead commands a maneuver or a weapons-state "
        "change. Do not call it for status requests or small talk."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "order": {
                "type": "string",
                "enum": list(ORDER_TYPES),
                "description": "The tactical order to execute.",
            },
            "heading_deg": {
                "type": ["number", "null"],
                "description": "Commanded heading in degrees for 'vector' orders.",
            },
            "altitude_m": {
                "type": ["number", "null"],
                "description": "Commanded altitude in meters, if the lead gave one.",
            },
            "speed_ms": {
                "type": ["number", "null"],
                "description": "Commanded speed in m/s, if the lead gave one.",
            },
            "station": {
                "type": ["string", "null"],
                "enum": sorted(FORMATIONS) + [None],
                "description": "Formation name for a 'formation' order.",
            },
            "side": {
                "type": ["string", "null"],
                "enum": ["left", "right", None],
                "description": "Which side of the lead to hold station on.",
            },
            "leash": {
                "type": ["string", "null"],
                "enum": list(LEASH_LEVELS) + [None],
                "description": "Supervised-autonomy level for a 'leash' order.",
            },
        },
        "required": ["order", "heading_deg", "altitude_m", "speed_ms",
                     "station", "side", "leash"],
        "additionalProperties": False,
    },
}


class WingmanAgent:
    """Claude-backed radio agent.

    A minimal manual tool loop is used instead of the SDK's beta tool
    runner so the agent stays testable with an injected fake client and
    has no beta dependency; the loop is two requests at most per
    transmission (tool call -> acknowledgment).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client=None,
        callsign: str = "Viper 2",
    ):
        if client is None:
            import anthropic  # deferred so offline mode needs no install

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.callsign = callsign
        self.messages: List[dict] = []

    # ------------------------------------------------------------------ #
    def radio(self, transmission: str, situation: dict) -> Tuple[str, List[TacticalOrder]]:
        """One radio exchange: returns (spoken reply, parsed orders)."""
        self.messages.append({
            "role": "user",
            "content": (
                f"<situation>{json.dumps(situation, ensure_ascii=False)}</situation>\n"
                f"Radio from flight lead: {transmission}"
            ),
        })

        orders: List[TacticalOrder] = []
        reply = ""
        for _ in range(3):  # tool round-trips are 1 in practice; bound anyway
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=[SET_ORDER_TOOL],
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},  # radio latency > depth
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    reply = block.text.strip()

            if not tool_uses:
                break

            results = []
            for tu in tool_uses:
                try:
                    order = TacticalOrder.from_tool_input(tu.input)
                    orders.append(order)
                    outcome = f"Order acknowledged by flight controls: {order.describe()}."
                except (KeyError, ValueError) as exc:
                    outcome = f"Order rejected: {exc}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": outcome,
                })
            self.messages.append({"role": "user", "content": results})

        self._trim_history()
        return reply or f"{self.callsign} copies.", orders

    def _trim_history(self) -> None:
        # Keep the last N lead transmissions (a "turn" spans user ->
        # assistant [-> tool_result -> assistant]); trim on user-text turns.
        starts = [
            i for i, m in enumerate(self.messages)
            if m["role"] == "user" and isinstance(m["content"], str)
        ]
        if len(starts) > MAX_HISTORY_TURNS:
            self.messages = self.messages[starts[len(starts) - MAX_HISTORY_TURNS]:]


# ---------------------------------------------------------------------- #
# Offline fallback: rule-based brevity parsing (no API needed)
# ---------------------------------------------------------------------- #
_BREVITY_RULES = [
    # weapons rules first: "weapons hold" must not match the anchor "hold"
    (r"weapons?\s*free|사격\s*(허가|자유)", "weapons_free"),
    (r"weapons?\s*(hold|safe)|사격\s*(중지|금지)", "weapons_hold"),
    (r"break\s*(left|port)|좌.*브레이크|브레이크.*좌", "break_left"),
    (r"break\s*(right|starboard)|우.*브레이크|브레이크.*우", "break_right"),
    (r"\bengage|\battack|\bcommit|교전|공격", "engage"),
    (r"\banchor|\bhold\b|\borbit|대기|선회", "anchor"),
    (r"\brtb\b|return to base|복귀|귀환", "rtb"),
    (r"\bstatus|\bcheck\s*in|\bpicture|상태|보고", "status"),
]

_ACK = {
    "engage": "2, engaging.",
    "anchor": "2 is anchored.",
    "vector": "2, on the vector.",
    "break_left": "2, breaking left!",
    "break_right": "2, breaking right!",
    "rtb": "2, returning to base.",
    "weapons_free": "2, weapons free.",
    "weapons_hold": "2, weapons hold.",
    "scout": "2, pushing ahead.",
    "status": "",
}

# formation station name -> matching regex (English + Korean)
_FORMATION_PATTERNS = [
    ("line_abreast",  r"line\s*abreast|횡대"),
    ("combat_spread", r"combat\s*spread|\bspread\b|전투\s*전개|컴뱃\s*스프레드"),
    ("wall",          r"\bwall\b|월\b"),
    ("fighting_wing", r"fighting\s*wing|파이팅\s*윙|밀집"),
    ("echelon",       r"echelon|사다리꼴|제형"),
    ("wedge",         r"\bwedge\b|쐐기|웨지"),
    ("trail",         r"\btrail\b|종대|트레일"),
]

_LEASH_PATTERNS = [
    ("loose", r"\bloose\b|weapons?\s*free\s*leash|풀어|루즈"),
    ("close", r"\bclose\s*leash|묶어|근접\s*(리쉬|편대\s*유지)|클로스"),
    ("tight", r"\btight\b|weapons?\s*tight|타이트"),
]


class BrevityParser:
    """Rule-based stand-in for :class:`WingmanAgent` (English + Korean)."""

    def radio(self, transmission: str, situation: dict) -> Tuple[str, List[TacticalOrder]]:
        text = transmission.lower()

        m = re.search(r"(?:heading|vector|헤딩)\s*(\d{1,3})", text)
        if m:
            heading = float(m.group(1)) % 360.0
            alt = re.search(r"(?:angels|altitude|고도)\s*(\d+)", text)
            altitude = float(alt.group(1)) if alt else None
            if altitude is not None and altitude < 100:  # "angels 5" style
                altitude *= 304.8
            order = TacticalOrder("vector", heading_deg=heading, altitude_m=altitude)
            return (f"2, coming to heading {heading:03.0f}.", [order])

        # leash change: check before weapons/formation so "weapons tight" wins
        for level, pattern in _LEASH_PATTERNS:
            if re.search(pattern, text):
                return (f"2, {level} leash.", [TacticalOrder("leash", leash=level)])

        # formation / rejoin / scout (CCA teaming)
        side = None
        if re.search(r"\bleft\b|좌측|왼", text):
            side = "left"
        elif re.search(r"\bright\b|우측|오른", text):
            side = "right"
        for name, pattern in _FORMATION_PATTERNS:
            if re.search(pattern, text):
                return (f"2, {name.replace('_', ' ')}.",
                        [TacticalOrder("formation", station=name, side=side)])
        if re.search(r"rejoin|form\s*up|편대\s*복귀|합류|재결합", text):
            return ("2, rejoining.", [TacticalOrder("rejoin", side=side)])
        if re.search(r"\bscout\b|\bpush\b|\bsweep\b|정찰|전진|스카웃", text):
            return (_ACK["scout"], [TacticalOrder("scout")])

        for pattern, name in _BREVITY_RULES:
            if re.search(pattern, text):
                if name == "status":
                    return (self._status_report(situation), [])
                return (_ACK[name], [TacticalOrder(name)])
        return ("2, say again?", [])

    @staticmethod
    def _status_report(situation: dict) -> str:
        bandit = situation.get("bandit")
        own = situation.get("own", {})
        mode = situation.get("mode")
        tail = f" {mode}." if mode in ("formation", "scout") else ""
        if bandit:
            return (
                f"2 has tally, {bandit.get('name', 'bandit')} at "
                f"{bandit.get('range_km', '?')} kilometers, own altitude "
                f"{own.get('altitude_m', 0):.0f} meters.{tail}"
            )
        return (f"2 is clean, altitude {own.get('altitude_m', 0):.0f} meters."
                f"{tail}")


def make_agent(model: str = DEFAULT_MODEL, offline: bool = False):
    """WingmanAgent when Claude is reachable, BrevityParser otherwise."""
    if offline:
        return BrevityParser()
    try:
        return WingmanAgent(model=model)
    except Exception as exc:  # no SDK installed / no credentials
        print(f"radio: Claude unavailable ({exc}); using offline brevity parser")
        return BrevityParser()
