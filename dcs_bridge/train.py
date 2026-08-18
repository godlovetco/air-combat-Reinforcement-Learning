"""Train the UCAV maneuver policy with numpy Q-learning.

Usage::

    python -m dcs_bridge.train --episodes 600 --out checkpoints/ucav_policy.npz

Replaces the legacy TF 1.x loop in ``main.py`` (which re-initialized the
network every gradient step) with a standard DQN-lite setup: experience
replay, a periodically synced target network, and an epsilon schedule.
The resulting ``.npz`` checkpoint is what ``dcs_bridge.run_pilot`` loads to
fly inside DCS World.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import time
from typing import Deque, Tuple

import numpy as np

from .policy import QNetwork
from .sim_env import UCAVSimEnv

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def parse_mixed_weights(spec):
    """'pursuit=3,straight=1,evasive=1' -> {'pursuit': 3.0, ...} (None passes)."""
    if not spec:
        return None
    weights = {}
    for part in spec.split(","):
        name, _, value = part.partition("=")
        name = name.strip()
        if not name or not value:
            raise ValueError(f"bad --mixed-weights entry {part!r}, expected name=weight")
        weights[name] = float(value)
    return weights


def selfplay_policy(args: argparse.Namespace, net: QNetwork, mixed_weights):
    """Frozen opponent network for self-play, or ``None`` if unused.

    Priority: an explicit ``--selfplay-init`` checkpoint, otherwise a frozen
    copy of the learner's starting weights (which, with ``--init``, is the
    shipped policy -- i.e. "beat the current champion").
    """
    wants = args.opponent == "selfplay" or bool(
        mixed_weights and mixed_weights.get("selfplay", 0.0) > 0
    ) or (args.eval_opponent == "selfplay")
    if not wants:
        return None
    init = getattr(args, "selfplay_init", None)
    if init:
        print(f"self-play opponent: {init}")
        return QNetwork.load(init)
    print("self-play opponent: frozen copy of the starting weights")
    return net.clone()


def train(args: argparse.Namespace) -> QNetwork:
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.init:
        net = QNetwork.load(args.init)  # warm-start / fine-tune from a checkpoint
        print(f"warm-starting from {args.init}")
    else:
        net = QNetwork(seed=args.seed)

    mixed_weights = parse_mixed_weights(getattr(args, "mixed_weights", None))
    bandit_policy = selfplay_policy(args, net, mixed_weights)
    env = UCAVSimEnv(
        max_steps=args.max_steps,
        randomize=not args.fixed_start,
        shaping=args.shaping,
        seed=args.seed,
        opponent=args.opponent,
        mixed_weights=mixed_weights,
        bandit_policy=bandit_policy,
    )
    target_net = net.clone()
    buffer: Deque[Transition] = collections.deque(maxlen=args.buffer_size)

    history = []
    outcomes = collections.deque(maxlen=50)
    best_score = -1.0
    t0 = time.time()

    for episode in range(1, args.episodes + 1):
        frac = min(1.0, episode / max(1, args.epsilon_decay_episodes))
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        obs = env.reset()
        ep_reward, ep_loss, updates = 0.0, 0.0, 0
        done = False
        info = {}

        while not done:
            action = net.act(obs, epsilon)
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs
            ep_reward += reward

            if len(buffer) >= args.batch_size:
                batch = random.sample(buffer, args.batch_size)
                states = np.stack([t[0] for t in batch])
                actions = [t[1] for t in batch]
                rewards = np.array([t[2] for t in batch])
                next_states = np.stack([t[3] for t in batch])
                dones = np.array([t[4] for t in batch], dtype=bool)

                next_q = target_net.forward(next_states).max(axis=1)
                targets = rewards + args.gamma * next_q * (~dones)
                ep_loss += net.train_batch(states, actions, targets, lr=args.lr)
                updates += 1

        if episode % args.target_sync == 0:
            target_net = net.clone()

        # Self-play curriculum: promote the learner to be its own opponent
        # every N episodes.  The opponent is always a *frozen* snapshot -- it
        # never trains mid-episode, which keeps the fight stationary enough
        # for the Q-targets to mean anything.
        if args.selfplay_refresh and env.bandit_policy is not None \
                and episode % args.selfplay_refresh == 0:
            env.set_bandit_policy(net.clone())

        outcomes.append(info.get("outcome"))
        win_rate = sum(1 for o in outcomes if o == "win") / len(outcomes)
        history.append(
            {
                "episode": episode,
                "steps": env.steps,
                "reward": round(ep_reward, 3),
                "loss": round(ep_loss / max(1, updates), 5),
                "epsilon": round(epsilon, 3),
                "outcome": info.get("outcome"),
                "win_rate_50": round(win_rate, 3),
            }
        )
        if episode % args.log_every == 0:
            h = history[-1]
            print(
                f"ep {h['episode']:4d}  steps {h['steps']:4d}  "
                f"reward {h['reward']:8.2f}  loss {h['loss']:8.5f}  "
                f"eps {h['epsilon']:.2f}  outcome {h['outcome']:>13}  "
                f"win50 {h['win_rate_50']:.2f}"
            )

        # DQN training oscillates; keep the best policy seen, not the last.
        if episode % args.eval_every == 0 and episode >= args.epsilon_decay_episodes // 2:
            win, conv = evaluate(net, episodes=args.select_episodes,
                                 seed=args.seed + episode,
                                 opponent=args.eval_opponent or args.opponent,
                                 bandit_policy=env.bandit_policy,
                                 mixed_weights=mixed_weights)
            score = win + 0.5 * conv
            if score > best_score:
                best_score = score
                net.save(args.out)
                print(
                    f"ep {episode}: new best policy saved "
                    f"(win {win:.2f}, conversion {conv:.2f})"
                )

    elapsed = time.time() - t0
    print(f"trained {args.episodes} episodes in {elapsed:.1f}s")

    if best_score < 0.0:
        net.save(args.out)
    print(f"checkpoint written to {args.out}")

    log_path = os.path.splitext(args.out)[0] + "_train_log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"training log written to {log_path}")
    return net


def evaluate(net: QNetwork, episodes: int = 20, seed: int = 1234,
             opponent: str = "straight", bandit_policy=None,
             mixed_weights=None):
    """Greedy evaluation.

    Returns ``(win_rate, conversion_rate)``.  A "conversion" ends the episode
    established in the bandit's rear hemisphere (own aspect < 30 deg, bandit
    aspect > 150 deg) -- the tracking-success criterion used for AI-pilot
    validation in DCS by Yoo/Kim/Shim (ICCAS 2021).  With equal aircraft
    speeds an outright gun-envelope win is only reachable from a well-timed
    intercept, so the conversion rate is the more informative metric.
    """
    from .geometry import situation

    needs_policy = opponent == "selfplay" or bool(
        mixed_weights and mixed_weights.get("selfplay", 0.0) > 0
    )
    if needs_policy and bandit_policy is None:
        bandit_policy = net.clone()  # mirror match against a frozen copy of itself
    env = UCAVSimEnv(randomize=True, shaping=0.0, seed=seed, opponent=opponent,
                     mixed_weights=mixed_weights, bandit_policy=bandit_policy)
    wins = 0
    conversions = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            obs, _, done, info = env.step(net.act(obs))
        if info.get("outcome") == "win":
            wins += 1
            conversions += 1
        else:
            feats = situation(env.pos_r, env.act_r, env.pos_b, env.act_b)
            if feats[0] < 30.0 and feats[1] > 150.0:
                conversions += 1
    return wins / episodes, conversions / episodes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--target-sync", type=int, default=10, metavar="EPISODES")
    p.add_argument("--shaping", type=float, default=0.05,
                   help="weight of the dense angular-advantage reward (0 = off)")
    p.add_argument("--opponent", default="straight",
                   choices=["straight", "pursuit", "evasive", "ace", "selfplay", "mixed"],
                   help="bandit behavior during training (mixed = randomized per episode)")
    p.add_argument("--eval-opponent", default=None,
                   choices=["straight", "pursuit", "evasive", "ace", "selfplay", "mixed"],
                   help="bandit behavior for periodic/final eval (default: same as --opponent)")
    p.add_argument("--mixed-weights", default=None, metavar="SPEC",
                   help="per-episode behavior weights for --opponent mixed, e.g. "
                        "'pursuit=3,straight=1,evasive=1' (rehearsal curriculum); "
                        "'selfplay' is only drawn if you weight it explicitly; "
                        "default = uniform over the scripted behaviors")
    p.add_argument("--selfplay-init", default=None, metavar="CHECKPOINT",
                   help="checkpoint that flies the bandit under --opponent selfplay "
                        "(default: a frozen copy of the learner's starting weights)")
    p.add_argument("--selfplay-refresh", type=int, default=0, metavar="EPISODES",
                   help="promote the learner to be its own frozen opponent every N "
                        "episodes (0 = keep the original self-play opponent)")
    p.add_argument("--fixed-start", action="store_true",
                   help="use the exact legacy head-on start instead of randomized geometry")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50, metavar="EPISODES",
                   help="periodic greedy eval; the best policy so far is what gets saved")
    p.add_argument("--select-episodes", type=int, default=12, metavar="N",
                   help="episodes per periodic eval used for best-checkpoint selection; "
                        "raise it when --opponent mixed draws many behaviors, since a "
                        "small sample makes the selection noisy")
    p.add_argument("--out", default="checkpoints/ucav_policy.npz")
    p.add_argument("--init", default=None, metavar="CHECKPOINT",
                   help="warm-start training from an existing checkpoint "
                        "(fine-tuning) instead of random weights")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="greedy evaluation episodes after training (0 = skip)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)
    if args.eval_episodes:
        net = QNetwork.load(args.out)  # the best policy is what was saved
        opp = args.eval_opponent or args.opponent
        bandit = QNetwork.load(args.selfplay_init) if args.selfplay_init else None
        win_rate, conversion_rate = evaluate(
            net, args.eval_episodes, opponent=opp, bandit_policy=bandit,
            mixed_weights=parse_mixed_weights(args.mixed_weights))
        print(
            f"greedy evaluation over {args.eval_episodes} episodes vs {opp}: "
            f"win rate {win_rate:.2f}, conversion rate {conversion_rate:.2f}"
        )


if __name__ == "__main__":
    main()
