"""Pure-numpy Q-network for the UCAV maneuver policy.

Architecture follows the legacy TensorFlow 1.x model in ``main.py``:

    input 72  ->  sigmoid 100  ->  sigmoid 30  ->  linear 9

The 9 outputs are Q-values for the 9 candidate maneuvers of
``geometry.ACTION_DELTAS``.  Implemented in numpy so the same file serves
training (backprop on the chosen action) and real-time inference inside the
DCS bridge with no ML-framework dependency.

The shape is a default, not a constraint: the ``energy`` action set needs
216 -> ... -> 27, so the layer sizes are constructor arguments and ``load``
reads them back out of the checkpoint's tensor shapes.  Old checkpoints keep
loading unchanged.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Sequence

import numpy as np

from .geometry import INPUT_DIM, NUM_ACTIONS

HIDDEN_SIZES = (100, 30)
LAYER_SIZES = (INPUT_DIM, *HIDDEN_SIZES, NUM_ACTIONS)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class QNetwork:
    """Three-layer MLP with sigmoid hidden units and a linear head."""

    def __init__(
        self,
        seed: Optional[int] = None,
        input_dim: int = INPUT_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden: Sequence[int] = HIDDEN_SIZES,
    ):
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.hidden = tuple(int(h) for h in hidden)
        layer_sizes = (self.input_dim, *self.hidden, self.num_actions)
        self.params = {}
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier/Glorot uniform
            self.params[f"W{i}"] = rng.uniform(-limit, limit, (fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros(fan_out)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray, want_cache: bool = False):
        """Q-values for a batch (N, input_dim) or a single (input_dim,) input."""
        squeeze = x.ndim == 1
        a0 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z1 = a0 @ self.params["W0"] + self.params["b0"]
        a1 = _sigmoid(z1)
        z2 = a1 @ self.params["W1"] + self.params["b1"]
        a2 = _sigmoid(z2)
        q = a2 @ self.params["W2"] + self.params["b2"]
        if want_cache:
            return q, (a0, a1, a2)
        return q[0] if squeeze else q

    def act(self, x: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy maneuver index for one input vector."""
        if epsilon > 0.0 and random.random() < epsilon:
            return random.randrange(self.num_actions)
        return int(np.argmax(self.forward(x)))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_batch(
        self,
        states: np.ndarray,
        actions: Sequence[int],
        targets: Sequence[float],
        lr: float = 0.05,
    ) -> float:
        """One SGD step of Q-learning regression.

        Only the Q-value of the taken action is pulled toward its TD target;
        gradients for the other heads are zero.  Returns the batch MSE.
        """
        states = np.atleast_2d(states)
        n = states.shape[0]
        actions = np.asarray(actions, dtype=int)
        targets = np.asarray(targets, dtype=np.float64)

        q, (a0, a1, a2) = self.forward(states, want_cache=True)
        picked = q[np.arange(n), actions]
        err = picked - targets
        loss = float(np.mean(err ** 2))

        dq = np.zeros_like(q)
        dq[np.arange(n), actions] = 2.0 * err / n

        p = self.params
        gW2 = a2.T @ dq
        gb2 = dq.sum(axis=0)
        da2 = dq @ p["W2"].T
        dz2 = da2 * a2 * (1.0 - a2)
        gW1 = a1.T @ dz2
        gb1 = dz2.sum(axis=0)
        da1 = dz2 @ p["W1"].T
        dz1 = da1 * a1 * (1.0 - a1)
        gW0 = a0.T @ dz1
        gb0 = dz1.sum(axis=0)

        for name, grad in (
            ("W2", gW2), ("b2", gb2),
            ("W1", gW1), ("b1", gb1),
            ("W0", gW0), ("b0", gb0),
        ):
            p[name] -= lr * grad
        return loss

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path: str) -> "QNetwork":
        """Load a checkpoint, taking its layer sizes from the stored tensors."""
        with np.load(path) as data:
            weights = sorted(k for k in data.files if k.startswith("W"))
            if not weights or weights != [f"W{i}" for i in range(len(weights))]:
                raise ValueError(f"checkpoint {path!r} has no usable weight tensors")
            shapes = [data[k].shape for k in weights]
            for i, shape in enumerate(shapes):
                if len(shape) != 2:
                    raise ValueError(f"checkpoint tensor 'W{i}' is not a matrix")
                if i and shape[0] != shapes[i - 1][1]:
                    raise ValueError(f"checkpoint tensor 'W{i}' does not chain from 'W{i-1}'")
            net = cls(
                seed=0,
                input_dim=shapes[0][0],
                num_actions=shapes[-1][1],
                hidden=[s[1] for s in shapes[:-1]],
            )
            for key in net.params:
                if key not in data:
                    raise ValueError(f"checkpoint {path!r} is missing tensor {key!r}")
                if data[key].shape != net.params[key].shape:
                    raise ValueError(
                        f"checkpoint tensor {key!r} has shape {data[key].shape}, "
                        f"expected {net.params[key].shape}"
                    )
                net.params[key] = data[key].astype(np.float64)
        return net

    def clone(self) -> "QNetwork":
        other = QNetwork(seed=0, input_dim=self.input_dim,
                         num_actions=self.num_actions, hidden=self.hidden)
        other.params = {k: v.copy() for k, v in self.params.items()}
        return other

    @property
    def action_set(self) -> str:
        """Name of the geometry action set this network was built for."""
        from .geometry import action_set_for
        return action_set_for(self.num_actions)
