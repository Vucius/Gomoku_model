import unittest

import numpy as np

from gomoku_model.features import (
    CHANNELS,
    center_distance_plane,
    coordconv_planes,
    edge_distance_plane,
    encode_position,
    legal_moves_plane,
    phase_plane,
)


class FeaturePlaneTests(unittest.TestCase):
    def test_edge_distance_is_zero_on_border_and_one_in_center(self) -> None:
        plane = edge_distance_plane(5, 5)

        self.assertEqual(float(plane[0, 0]), 0.0)
        self.assertEqual(float(plane[0, 2]), 0.0)
        self.assertEqual(float(plane[2, 2]), 1.0)
        self.assertEqual(float(plane[1, 1]), 0.5)

    def test_coordconv_planes_are_normalized(self) -> None:
        x_plane, y_plane = coordconv_planes(3, 5)

        np.testing.assert_allclose(x_plane[0], [-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(y_plane[:, 0], [-1.0, 0.0, 1.0])

    def test_center_distance_peaks_at_center(self) -> None:
        plane = center_distance_plane(5, 5)

        self.assertAlmostEqual(float(plane[2, 2]), 1.0)
        self.assertLess(float(plane[0, 0]), float(plane[1, 1]))

    def test_legal_moves_plane_marks_empty_cells(self) -> None:
        board = np.array([[1, 0], [-1, 0]])

        np.testing.assert_array_equal(legal_moves_plane(board), np.array([[0, 1], [0, 1]], dtype=np.float32))

    def test_phase_plane_clamps_to_one(self) -> None:
        plane = phase_plane(300, 15, 15)

        self.assertEqual(float(plane[0, 0]), 1.0)

    def test_encode_position_uses_current_player_perspective(self) -> None:
        board = np.array(
            [
                [1, 0, -1],
                [0, -1, 0],
                [1, 0, 0],
            ]
        )

        encoded = encode_position(board, current_player=-1, move_count=4)

        self.assertEqual(encoded.shape, (len(CHANNELS), 3, 3))
        np.testing.assert_array_equal(
            encoded[0],
            np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            encoded[1],
            np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32),
        )
        self.assertAlmostEqual(float(encoded[-1, 0, 0]), 4 / 9)

    def test_phase_channel_requires_move_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "move_count"):
            encode_position(np.zeros((3, 3)), current_player=1)


if __name__ == "__main__":
    unittest.main()
