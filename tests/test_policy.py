import os
import tempfile
import unittest

import numpy as np

from dcs_bridge.geometry import INPUT_DIM, NUM_ACTIONS
from dcs_bridge.policy import QNetwork


class PolicyTest(unittest.TestCase):
    def test_forward_shapes(self):
        net = QNetwork(seed=1)
        single = net.forward(np.zeros(INPUT_DIM))
        self.assertEqual(single.shape, (NUM_ACTIONS,))
        batch = net.forward(np.zeros((5, INPUT_DIM)))
        self.assertEqual(batch.shape, (5, NUM_ACTIONS))

    def test_training_reduces_loss(self):
        rng = np.random.default_rng(0)
        net = QNetwork(seed=2)
        states = rng.normal(size=(64, INPUT_DIM))
        actions = rng.integers(0, NUM_ACTIONS, size=64)
        targets = rng.normal(size=64)

        first = net.train_batch(states, actions, targets, lr=0.5)
        for _ in range(200):
            last = net.train_batch(states, actions, targets, lr=0.5)
        self.assertLess(last, first * 0.5,
                        f"loss did not drop: first={first}, last={last}")

    def test_gradient_isolated_to_taken_action(self):
        """Untaken actions' Q-values should barely move on one update."""
        net = QNetwork(seed=3)
        x = np.ones(INPUT_DIM) * 0.1
        before = net.forward(x).copy()
        net.train_batch(x[None, :], [4], [before[4] + 5.0], lr=1.0)
        after = net.forward(x)
        moved = np.abs(after - before)
        self.assertGreater(moved[4], moved[[i for i in range(9) if i != 4]].max())

    def test_save_load_roundtrip(self):
        net = QNetwork(seed=4)
        x = np.linspace(-1, 1, INPUT_DIM)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policy.npz")
            net.save(path)
            loaded = QNetwork.load(path)
        np.testing.assert_allclose(net.forward(x), loaded.forward(x))

    def test_epsilon_greedy_bounds(self):
        net = QNetwork(seed=5)
        x = np.zeros(INPUT_DIM)
        for _ in range(50):
            self.assertIn(net.act(x, epsilon=1.0), range(NUM_ACTIONS))
        greedy = net.act(x, epsilon=0.0)
        self.assertEqual(greedy, int(np.argmax(net.forward(x))))


if __name__ == "__main__":
    unittest.main()
