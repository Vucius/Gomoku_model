import unittest

import numpy as np

from gomoku_model.targets import (
    label_smooth_policy,
    legal_uniform_policy,
    normalize_policy,
    policy_from_move,
    soften_visit_counts,
)


class TargetTests(unittest.TestCase):
    def test_policy_from_move_uses_xy_coord(self) -> None:
        policy = policy_from_move((2, 1), 3, 4)

        self.assertEqual(policy.shape, (3, 4))
        self.assertEqual(float(policy[1, 2]), 1.0)
        self.assertEqual(float(policy.sum()), 1.0)

    def test_legal_uniform_policy_masks_illegal_cells(self) -> None:
        legal = np.array([[True, False], [True, True]])
        policy = legal_uniform_policy(legal)

        np.testing.assert_allclose(policy, [[1 / 3, 0], [1 / 3, 1 / 3]])

    def test_normalize_policy_applies_legal_mask(self) -> None:
        policy = np.array([[1.0, 99.0], [3.0, 0.0]])
        legal = np.array([[True, False], [True, True]])

        np.testing.assert_allclose(normalize_policy(policy, legal), [[0.25, 0.0], [0.75, 0.0]])

    def test_label_smooth_policy_blends_with_uniform_legal(self) -> None:
        policy = np.array([[1.0, 0.0], [0.0, 0.0]])
        legal = np.array([[True, False], [True, True]])

        smoothed = label_smooth_policy(policy, legal, epsilon=0.3)

        np.testing.assert_allclose(smoothed, [[0.8, 0.0], [0.1, 0.1]], rtol=1e-6)
        self.assertAlmostEqual(float(smoothed.sum()), 1.0)

    def test_soften_visit_counts_respects_temperature(self) -> None:
        visits = np.array([[100.0, 25.0], [0.0, 0.0]])
        softened = soften_visit_counts(visits, temperature=2.0)

        np.testing.assert_allclose(softened, [[2 / 3, 1 / 3], [0.0, 0.0]], rtol=1e-6)

    def test_zero_visit_counts_fall_back_to_uniform_legal(self) -> None:
        legal = np.array([[True, False], [False, True]])
        softened = soften_visit_counts(np.zeros((2, 2)), temperature=1.0, legal_mask=legal)

        np.testing.assert_allclose(softened, [[0.5, 0.0], [0.0, 0.5]])


if __name__ == "__main__":
    unittest.main()
