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


def train(args: argparse.Namespace) -> QNetwork:
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = UCAVSimEnv(
        max_steps=args.max_steps,
        randomize=not args.fixed_start,
        shaping=args.shaping,
        seed=args.seed,
        opponent=args.opponent,
    )
    if args.init:
        net = QNetwork.load(args.init)  # warm-start / fine-tune from a checkpoint
        print(f"warm-starting from {args.init}")
    else:
        net = QNetwork(seed=args.seed)
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
            win, conv = evaluate(net, episodes=12, seed=args.seed + episode,
                                 opponent=args.eval_opponent or args.opponent)
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
             opponent: str = "straight"):
    """Greedy evaluation.

    Returns ``(win_rate, conversion_rate)``.  A "conversion" ends the episode
    established in the bandit's rear hemisphere (own aspect < 30 deg, bandit
    aspect > 150 deg) -- the tracking-success criterion used for AI-pilot
    validation in DCS by Yoo/Kim/Shim (ICCAS 2021).  With equal aircraft
    speeds an outright gun-envelope win is only reachable from a well-timed
    intercept, so the conversion rate is the more informative metric.
    """
    from .geometry import situation

    env = UCAVSimEnv(randomize=True, shaping=0.0, seed=seed, opponent=opponent)
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
                   choices=["straight", "pursuit", "evasive", "mixed"],
                   help="bandit behavior during training (mixed = randomized per episode)")
    p.add_argument("--eval-opponent", default=None,
                   choices=["straight", "pursuit", "evasive", "mixed"],
                   help="bandit behavior for periodic/final eval (default: same as --opponent)")
    p.add_argument("--fixed-start", action="store_true",
                   help="use the exact legacy head-on start instead of randomized geometry")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50, metavar="EPISODES",
                   help="periodic greedy eval; the best policy so far is what gets saved")
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
        win_rate, conversion_rate = evaluate(net, args.eval_episodes, opponent=opp)
        print(
            f"greedy evaluation over {args.eval_episodes} episodes vs {opp}: "
            f"win rate {win_rate:.2f}, conversion rate {conversion_rate:.2f}"
        )


if __name__ == "__main__":
    main()
