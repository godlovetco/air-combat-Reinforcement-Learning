"""Beyond-visual-range 1v1 environment.

The gun-fight environment in :mod:`dcs_bridge.sim_env` starts 10 km apart and
ends when someone converts to the other's six.  A BVR fight starts at 70 km and
is over before either aircraft ever points at the other for long: it is decided
by radar, weapon envelope and missile timeline, which live in
:mod:`dcs_bridge.bvr`.

The split of responsibility here is deliberate.  *When* to shoot is a rule --
take the shot when the target is locked and inside the envelope, prefer the
no-escape zone -- because that part is well understood and does not need to be
learned.  *How to fly* is the hard part and is what the policy learns: how far
you can crank off the target while still supporting your own missile, when to
commit to the notch against theirs, and when the fight is lost and you drag.

Observation = the usual per-candidate situation vectors, plus a
:data:`BVR_STATE_DIM`-wide block describing the missile timeline, so a policy
can see what the geometry alone cannot: who is locked, what is in the air, and
how much support your own shot still needs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from . import bvr
from . import geometry as geo
from .sim_env import MIXED_BEHAVIORS, MIXED_POOL, OPPONENTS

ARENA_XY = 400_000.0      # m, a BVR fight needs room to drag
ARENA_Z = 15_000.0        # m
FLOOR_Z = 300.0           # m

START_RANGE = 70_000.0    # m between the two at the merge start
START_ALT = 8_000.0       # m
START_SPEED = 250.0       # m/s

MERGE_RANGE = 9_000.0     # m at which a BVR fight has become a dogfight
CRANK_DEG = 50.0          # off-boresight a supporting shooter can hold
NOTCH_ERROR_DEG = 12.0    # how precisely a scripted bandit can hold the beam

BVR_STATE_DIM = 15
OUTCOMES = ("win", "loss", "merge", "out_of_bounds", "bandit_departed", "timeout")


def observation_dim(action_set: str = geo.DEFAULT_ACTION_SET) -> int:
    """Network input width for a BVR policy on this action set."""
    num_actions, base = geo.action_set_dims(action_set)
    return base + BVR_STATE_DIM


class BVRSimEnv:
    """Gym-style 1v1 BVR environment (same call surface as ``UCAVSimEnv``)."""

    def __init__(
        self,
        max_steps: int = 600,
        dt: float = 1.0,
        randomize: bool = True,
        shaping: float = 0.05,
        seed: Optional[int] = None,
        opponent: str = "ace",
        mixed_weights: Optional[dict] = None,
        bandit_policy=None,
        action_set: str = geo.DEFAULT_ACTION_SET,
        missiles: int = 4,
        spec: bvr.MissileSpec = bvr.DEFAULT_MISSILE,
        radar: Optional[bvr.Radar] = None,
        bandit_radar: Optional[bvr.Radar] = None,
        bandit_spec: Optional[bvr.MissileSpec] = None,
        notch_error: float = NOTCH_ERROR_DEG,
        start_range: float = START_RANGE,
    ):
        if action_set not in geo.ACTION_SETS:
            raise ValueError(
                f"unknown action set {action_set!r}, expected {tuple(geo.ACTION_SETS)}"
            )
        if opponent not in OPPONENTS:
            raise ValueError(f"unknown opponent {opponent!r}, expected {OPPONENTS}")
        if mixed_weights is not None:
            bad = set(mixed_weights) - set(MIXED_POOL)
            if bad:
                raise ValueError(f"unknown mixed_weights behaviors: {sorted(bad)}")
            if not any(w > 0 for w in mixed_weights.values()):
                raise ValueError("mixed_weights must contain a positive weight")
        wants_selfplay = opponent == "selfplay" or bool(
            mixed_weights and mixed_weights.get("selfplay", 0.0) > 0
        )
        if wants_selfplay and bandit_policy is None:
            raise ValueError("opponent 'selfplay' requires a bandit_policy")
        if missiles < 1:
            raise ValueError("each side needs at least one missile")

        self.max_steps = max_steps
        self.dt = dt
        self.randomize = randomize
        self.shaping = shaping
        self.opponent = opponent
        self.mixed_weights = mixed_weights
        self.bandit_policy = bandit_policy
        self.action_set = action_set
        self.loadout = missiles
        self.spec = spec
        self.bandit_spec = bandit_spec or spec
        # Radars are per-side so an AESA-versus-mechanical matchup -- the whole
        # point of carrying two array types -- is expressible.
        self.radar = radar or bvr.Radar.mechanical()
        self.bandit_radar = bandit_radar or bvr.Radar.mechanical()
        # How sloppily the scripted bandit flies its notch -- an opponent-skill
        # dial.  At 0 it beams perfectly and no Doppler radar of either type
        # can hold it; widen it and the arrays start to separate.
        self.notch_error = abs(notch_error)
        self.start_range = start_range
        import random as _random
        self.rng = _random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        mid = ARENA_XY / 2.0
        self.pos_r = [mid, mid - self.start_range / 2.0, START_ALT]
        self.pos_b = [mid, mid + self.start_range / 2.0, START_ALT]
        self.act_r = [START_SPEED, 0.0, 0.0]      # north, at each other
        self.act_b = [START_SPEED, 0.0, 180.0]

        if self.randomize:
            r = self.rng
            self.pos_b[0] += r.uniform(-15_000.0, 15_000.0)
            self.pos_b[1] += r.uniform(-10_000.0, 10_000.0)
            self.pos_b[2] += r.uniform(-3_000.0, 3_000.0)
            self.pos_r[2] += r.uniform(-2_000.0, 2_000.0)
            self.act_b[2] = geo.wrap_heading(180.0 + r.uniform(-25.0, 25.0))
            self.act_r[2] = geo.wrap_heading(r.uniform(-25.0, 25.0))

        if self.opponent == "mixed":
            if self.mixed_weights:
                behaviors = MIXED_POOL
                weights = [self.mixed_weights.get(b, 0.0) for b in behaviors]
                self._episode_opponent = self.rng.choices(behaviors, weights=weights)[0]
            else:
                self._episode_opponent = self.rng.choice(MIXED_BEHAVIORS)
        else:
            self._episode_opponent = self.opponent

        # A scripted bandit does not hold a perfect 90 degree beam, and the
        # difference matters: the mechanical Doppler gate is about +/-11 deg
        # wide at these speeds and an AESA's is about +/-7, so how precisely
        # the notch is flown is exactly what separates the two arrays.  A
        # bandit that beams perfectly every time defeats both and makes every
        # engagement a merge, which is not a fight -- it is an artifact.
        self._notch_error = self.rng.uniform(-self.notch_error, self.notch_error)

        self.missiles_r = self.loadout
        self.missiles_b = self.loadout
        self.in_flight: List[bvr.Missile] = []
        self.steps = 0
        self.done = False
        self._last_shot = {"agent": -1e9, "bandit": -1e9}
        # Both sides start the merge already tracking each other if the
        # geometry supports it; after that a lost track has to be re-acquired,
        # which is what makes notching a mechanical radar expensive.
        self._track = {
            "agent": {"locked": self._raw_lock("agent"), "timer": 0.0},
            "bandit": {"locked": self._raw_lock("bandit"), "timer": 0.0},
        }
        return self._obs()

    # ------------------------------------------------------------------ #
    # Radar / weapons bookkeeping
    # ------------------------------------------------------------------ #
    def radar_of(self, who: str) -> bvr.Radar:
        return self.radar if who == "agent" else self.bandit_radar

    def spec_of(self, who: str) -> bvr.MissileSpec:
        return self.spec if who == "agent" else self.bandit_spec

    def _raw_lock(self, who: str) -> bool:
        """Line of sight right now, before the re-acquisition delay."""
        if who == "agent":
            return self.radar.can_see(self.pos_r, self.act_r, self.pos_b, self.act_b)
        return self.bandit_radar.can_see(self.pos_b, self.act_b,
                                         self.pos_r, self.act_r)

    def _update_tracks(self) -> None:
        """Advance each side's track state, honoring its re-acquisition time."""
        for who in ("agent", "bandit"):
            st = self._track[who]
            if not self._raw_lock(who):
                st["locked"] = False
                st["timer"] = 0.0
            elif not st["locked"]:
                st["timer"] += self.dt
                if st["timer"] >= self.radar_of(who).reacquire_time:
                    st["locked"] = True
                    st["timer"] = 0.0

    def _lock(self, who: str) -> bool:
        return self._track[who]["locked"]

    def _spiked(self, who: str) -> bool:
        """Whether ``who``'s RWR registers the other side's radar.

        An LPI array can hold a track from well outside the range at which its
        emissions light up the target's warning receiver, so a locked aircraft
        does not necessarily know it.
        """
        other = "bandit" if who == "agent" else "agent"
        if not self._lock(other):
            return False
        emitter = self.radar_of(other)
        pos_emitter = self.pos_b if other == "bandit" else self.pos_r
        pos_target = self.pos_r if who == "agent" else self.pos_b
        return emitter.is_detected_by_rwr(pos_emitter, pos_target)

    def _shots(self, shooter: str) -> List[bvr.Missile]:
        return [m for m in self.in_flight if m.shooter == shooter and m.alive]

    def _threats(self, to: str) -> List[bvr.Missile]:
        other = "bandit" if to == "agent" else "agent"
        return self._shots(other)

    def _authority(self, shooter: str) -> dict:
        if shooter == "agent":
            return bvr.launch_authority(self.spec, self.pos_r, self.act_r,
                                        self.pos_b, self.act_b)  # noqa: E501
        return bvr.launch_authority(self.bandit_spec, self.pos_b, self.act_b,
                                    self.pos_r, self.act_r)

    def _try_launch(self, shooter: str) -> bool:
        """Rule-based shot: locked, in the envelope, and not double-shooting."""
        left = self.missiles_r if shooter == "agent" else self.missiles_b
        if left <= 0 or not self._lock(shooter):
            return False
        if self.steps - self._last_shot[shooter] < 8:   # trigger discipline
            return False
        auth = self._authority(shooter)
        supporting = [m for m in self._shots(shooter) if not m.active]
        if len(supporting) >= self.radar_of(shooter).simultaneous_tracks:
            return False   # the radar cannot guide another one
        if not auth["in_envelope"]:
            return False
        # Outside the no-escape zone a shot is a "maddog" that mostly forces a
        # defensive reaction; take it only if nothing is already in the air.
        if not auth["in_nez"] and supporting:
            return False
        pos, act = ((self.pos_r, self.act_r) if shooter == "agent"
                    else (self.pos_b, self.act_b))
        self.in_flight.append(
            bvr.Missile.launch(self.spec_of(shooter), shooter, pos, act))
        self._last_shot[shooter] = self.steps
        if shooter == "agent":
            self.missiles_r -= 1
        else:
            self.missiles_b -= 1
        return True

    def _step_missiles(self) -> Optional[str]:
        """Advance every live missile.  Returns "win"/"loss" on a hit."""
        result = None
        for m in list(self.in_flight):
            if not m.alive:
                continue
            if m.shooter == "agent":
                tgt_pos, tgt_act, supported = self.pos_b, self.act_b, self._lock("agent")
            else:
                tgt_pos, tgt_act, supported = self.pos_r, self.act_r, self._lock("bandit")
            outcome = m.step(tgt_pos, tgt_act, supported=supported, dt=self.dt)
            if outcome == "hit":
                result = "win" if m.shooter == "agent" else "loss"
        self.in_flight = [m for m in self.in_flight if m.alive]
        return result

    # ------------------------------------------------------------------ #
    # Bandit
    # ------------------------------------------------------------------ #
    def _update_bandit(self) -> None:
        behavior = getattr(self, "_episode_opponent", "ace")
        if behavior == "straight":
            return
        if behavior == "selfplay":
            obs = self._obs_for(self.pos_b, self.act_b, self.pos_r, self.act_r,
                                "bandit")
            action = self.bandit_policy.act(obs)
            self.act_b = geo.candidate_actions(
                *self.act_b, action_set=self.action_set, dt=self.dt
            )[action]
            return

        hot = geo.wrap_heading(math.degrees(math.atan2(
            self.pos_r[0] - self.pos_b[0], self.pos_r[1] - self.pos_b[1])))
        threats = [m for m in self._threats("bandit") if m.active]
        supporting = [m for m in self._shots("bandit") if not m.active]

        if behavior == "pursuit":
            desired_psi = hot          # banzai: straight at us the whole way
        elif behavior == "evasive":
            desired_psi = self._beam_heading() if threats else hot
        else:  # "ace" -- fly the BVR timeline off the RWR, not off hindsight
            if threats:
                desired_psi = self._beam_heading()      # defeat what is in the air
            elif self._spiked("bandit") and self._inside_their_wez("bandit"):
                # Spiked from inside the shooter's envelope: a shot may already
                # be in the air, and notching *now* -- before the seeker goes
                # active -- breaks the mid-course lock and kills it outright.
                # This is the only reason an LPI array is worth anything: an
                # opponent that reacts only to missiles whose seekers are
                # already emitting cannot be denied its warning, so it can
                # never be caught committing.
                desired_psi = self._beam_heading()
            elif supporting:
                desired_psi = geo.wrap_heading(hot + CRANK_DEG)  # crank and support
            else:
                desired_psi = hot                       # commit

        v_b, gamma_b, psi_b = self.act_b
        turn = max(-10.0, min(10.0, geo.heading_error(desired_psi, psi_b)))
        psi_b = geo.wrap_heading(psi_b + turn)
        up_limit = (geo.GAMMA_LIMIT_DEG if self.action_set == "legacy"
                    else geo.max_climb_angle(v_b))
        gamma_b = max(-geo.GAMMA_LIMIT_DEG, min(up_limit, gamma_b * 0.8))
        if self.action_set != "legacy":
            v_b = geo.energy_step(v_b, gamma_b, turn, 1.0, self.dt)
        self.act_b = [v_b, gamma_b, psi_b]

    def _inside_their_wez(self, who: str) -> bool:
        """Whether ``who`` is inside the *other* side's weapon envelope."""
        other = "bandit" if who == "agent" else "agent"
        return self._authority(other)["in_envelope"]

    def _beam_heading(self) -> float:
        """Heading that puts the threat on the bandit's 3/9 line -- the notch.

        The threat is the incoming missile when there is one, not the aircraft
        that fired it.  Those bearings diverge as the round closes, and a pilot
        who keeps beaming the shooter slides out of the seeker's Doppler gate
        while believing they are still notching.
        """
        threats = [m for m in self._threats("bandit") if m.active]
        aim = threats[0].pos if threats else self.pos_r
        hot = geo.wrap_heading(math.degrees(math.atan2(
            aim[0] - self.pos_b[0], aim[1] - self.pos_b[1])))
        err = getattr(self, "_notch_error", 0.0)
        left = geo.wrap_heading(hot - 90.0 + err)
        right = geo.wrap_heading(hot + 90.0 + err)
        psi_b = self.act_b[2]
        return (left if abs(geo.heading_error(left, psi_b))
                <= abs(geo.heading_error(right, psi_b)) else right)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _bvr_block(self, own_pos, own_act, tgt_pos, tgt_act, who: str) -> np.ndarray:
        other = "bandit" if who == "agent" else "agent"
        left = self.missiles_r if who == "agent" else self.missiles_b
        left_other = self.missiles_b if who == "agent" else self.missiles_r
        mine = self._shots(who)
        threats = self._shots(other)
        auth = bvr.launch_authority(self.spec_of(who), own_pos, own_act,
                                    tgt_pos, tgt_act)

        supported = [m for m in mine if not m.active]
        owed = min(1.0, supported[0].time_to_active(tgt_pos) / 60.0) if supported else 0.0
        threat_active = any(m.active for m in threats)
        if threats:
            _los, td = bvr.line_of_sight(threats[0].pos, own_pos)
            tti = min(1.0, td / max(threats[0].speed, 1.0) / 60.0)
        else:
            tti = 1.0
        rng_ratio = min(2.0, auth["range"] / max(auth["rmax"], 1.0)) / 2.0

        # Where the threat actually is, and how deep in its notch we are.
        # Without these the block says "a missile is inbound, N seconds out"
        # and nothing about its bearing -- and a notch is a maneuver *relative
        # to the threat*, so a policy given only the first two facts cannot
        # learn to fly one. That omission, not the hyper-parameters, is why the
        # first trained BVR policies lost to a twenty-line scripted timeline.
        if threats:
            t_pos = threats[0].pos
            bearing = geo.wrap_heading(math.degrees(math.atan2(
                t_pos[0] - own_pos[0], t_pos[1] - own_pos[1])))
            off = math.radians(geo.heading_error(bearing, own_act[2]))
            threat_sin, threat_cos = math.sin(off), math.cos(off)
            # Our own velocity along the line of sight to the threat,
            # normalized and signed so +1 is flying straight at it and -1 is
            # running from it.  Zero is exactly the Doppler notch -- the
            # seeker judges the same quantity from its end, where only the
            # magnitude matters.
            notch_depth = max(-1.0, min(1.0, bvr.radial_speed(
                own_pos, t_pos, own_act) / max(own_act[0], 1.0)))
        else:
            threat_sin = threat_cos = notch_depth = 0.0

        return np.array([
            1.0 if self._lock(who) else 0.0,
            1.0 if self._spiked(who) else 0.0,
            left / max(1, self.loadout),
            left_other / max(1, self.loadout),
            1.0 if mine else 0.0,
            owed,
            1.0 if threats else 0.0,
            1.0 if threat_active else 0.0,
            tti,
            1.0 if auth["in_envelope"] else 0.0,
            1.0 if auth["in_nez"] else 0.0,
            rng_ratio,
            threat_sin,
            threat_cos,
            notch_depth,
        ])

    def _obs_for(self, own_pos, own_act, tgt_pos, tgt_act, who: str) -> np.ndarray:
        base = geo.build_network_input(own_pos, own_act, tgt_pos, tgt_act, self.dt,
                                       action_set=self.action_set)
        return np.concatenate([base,
                               self._bvr_block(own_pos, own_act, tgt_pos, tgt_act, who)])

    def _obs(self) -> np.ndarray:
        return self._obs_for(self.pos_r, self.act_r, self.pos_b, self.act_b, "agent")

    # ------------------------------------------------------------------ #
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        cands = geo.candidate_actions(*self.act_r, action_set=self.action_set,
                                      dt=self.dt)
        self.act_r = cands[action_idx]
        self._update_bandit()
        self.pos_r = geo.step_point_mass(self.pos_r, self.act_r, self.dt)
        self.pos_b = geo.step_point_mass(self.pos_b, self.act_b, self.dt)
        self.steps += 1

        self._update_tracks()
        self._try_launch("agent")
        self._try_launch("bandit")
        hit = self._step_missiles()

        _los, d = bvr.line_of_sight(self.pos_r, self.pos_b)
        reward = 0.0
        info = {"outcome": None, "d": d,
                "missiles_r": self.missiles_r, "missiles_b": self.missiles_b,
                "in_flight": len(self.in_flight)}

        if self.shaping:
            reward += self.shaping * self._shaping_term()

        if hit == "win":
            reward += 10.0
            self.done = True
            info["outcome"] = "win"
        elif hit == "loss":
            reward -= 10.0
            self.done = True
            info["outcome"] = "loss"
        elif (self.missiles_r == 0 and self.missiles_b == 0
              and not self.in_flight and d < MERGE_RANGE):
            # Both sides Winchester and inside visual range: this is no longer a
            # BVR problem, and pretending to score it as one would be dishonest.
            self.done = True
            info["outcome"] = "merge"
        else:
            who = self._departed()
            if who == "agent":
                reward -= 5.0
                self.done = True
                info["outcome"] = "out_of_bounds"
            elif who == "bandit":
                self.done = True
                info["outcome"] = "bandit_departed"
            elif self.steps >= self.max_steps:
                reward -= 2.0
                self.done = True
                info["outcome"] = "timeout"

        return self._obs(), reward, self.done, info

    def _shaping_term(self) -> float:
        """Support your shot; defeat theirs.

        Two terms, both of which a BVR pilot would recognize: while your own
        missile still needs the radar, holding the lock is worth something and
        losing it throws the shot away; while a missile is guiding on you,
        being in its notch is worth something and being tracked is not.
        """
        r = 0.0
        if any(not m.active for m in self._shots("agent")):
            r += 1.0 if self._lock("agent") else -1.0
        threats = [m for m in self._threats("agent") if m.active]
        if threats:
            tracked = any(m.is_tracking(self.pos_r, self.act_r) for m in threats)
            r += -1.0 if tracked else 1.0
        return r

    # ------------------------------------------------------------------ #
    @staticmethod
    def _outside(pos) -> bool:
        if not (0.0 < pos[0] < ARENA_XY and 0.0 < pos[1] < ARENA_XY):
            return True
        return not (FLOOR_Z < pos[2] < ARENA_Z)

    def _departed(self) -> Optional[str]:
        if self._outside(self.pos_r):
            return "agent"
        if self._outside(self.pos_b):
            return "bandit"
        return None

    def set_bandit_policy(self, net) -> None:
        self.bandit_policy = net

    def trajectory_header(self) -> List[str]:
        return ["x_r", "y_r", "z_r", "x_b", "y_b", "z_b"]

    def trajectory_row(self) -> List[float]:
        return [*self.pos_r, *self.pos_b]
