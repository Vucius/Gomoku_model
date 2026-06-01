import unittest

import numpy as np

from gomoku_model.sampling import (
    EDGE_BUCKETS,
    FullBoardDataset,
    bucket_indices,
    edge_bucket_for_coord,
    edge_distance_for_coord,
    sample_balanced_indices,
)


class SamplingTests(unittest.TestCase):
    def test_edge_distance_for_xy_coord(self) -> None:
        self.assertEqual(edge_distance_for_coord((0, 0), 15, 15), 0)
        self.assertEqual(edge_distance_for_coord((1, 7), 15, 15), 1)
        self.assertEqual(edge_distance_for_coord((2, 12), 15, 15), 2)
        self.assertEqual(edge_distance_for_coord((7, 7), 15, 15), 7)

    def test_edge_bucket_for_coord(self) -> None:
        self.assertEqual(edge_bucket_for_coord((0, 14), 15, 15), "edge_0")
        self.assertEqual(edge_bucket_for_coord((1, 3), 15, 15), "edge_1")
        self.assertEqual(edge_bucket_for_coord((2, 3), 15, 15), "edge_2")
        self.assertEqual(edge_bucket_for_coord((3, 3), 15, 15), "center")

    def test_bucket_indices_returns_all_buckets(self) -> None:
        coords = np.array([[0, 0], [1, 5], [2, 5], [7, 7]])
        grouped = bucket_indices(coords, 15, 15)

        self.assertEqual(tuple(grouped), EDGE_BUCKETS)
        self.assertEqual(grouped["edge_0"].tolist(), [0])
        self.assertEqual(grouped["edge_1"].tolist(), [1])
        self.assertEqual(grouped["edge_2"].tolist(), [2])
        self.assertEqual(grouped["center"].tolist(), [3])

    def test_sample_balanced_indices_uses_requested_weights(self) -> None:
        coords = np.array([[0, 0], [1, 5], [2, 5], [7, 7]])
        sampled = sample_balanced_indices(
            coords,
            15,
            15,
            10,
            bucket_weights={"edge_0": 0.2, "edge_1": 0.2, "edge_2": 0.2, "center": 0.4},
            rng=np.random.default_rng(1),
        )
        counts = np.bincount(sampled, minlength=4)

        self.assertEqual(counts.tolist(), [2, 2, 2, 4])

    def test_full_board_dataset_encodes_and_samples_batch(self) -> None:
        boards = np.zeros((4, 15, 15), dtype=np.int8)
        boards[0, 7, 7] = 1
        boards[1, 1, 1] = -1
        coords = np.array([[0, 0], [1, 5], [2, 5], [7, 7]])
        players = np.array([1, -1, 1, -1])
        dataset = FullBoardDataset(boards, coords, players)

        encoded = dataset.encode(0)
        self.assertEqual(encoded.shape, (8, 15, 15))

        indices, features, moves, batch_players = dataset.sample_balanced_batch(
            4,
            bucket_weights={"edge_0": 0.25, "edge_1": 0.25, "edge_2": 0.25, "center": 0.25},
            rng=np.random.default_rng(2),
        )

        self.assertEqual(indices.shape, (4,))
        self.assertEqual(features.shape, (4, 8, 15, 15))
        self.assertEqual(moves.shape, (4, 2))
        self.assertEqual(batch_players.shape, (4,))


if __name__ == "__main__":
    unittest.main()
