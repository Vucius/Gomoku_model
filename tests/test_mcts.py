import unittest
import numpy as np
import torch
from src.mcts import TreeNode, MCTS

class DummyModel:
    def __init__(self, board_size=15):
        self.board_size = board_size
    def eval(self):
        pass
    def __call__(self, x):
        batch_size = x.shape[0]
        # Return uniform logits and constant value 0.5
        policy_logits = torch.zeros(batch_size, self.board_size * self.board_size)
        value = torch.full((batch_size, 1), 0.5)
        return {
            "policy_logits": policy_logits,
            "value": value
        }

class MCTSTests(unittest.TestCase):
    def test_tree_node_selection(self):
        root = TreeNode()
        root.expand({0: 0.8, 1: 0.2})
        self.assertEqual(len(root.children), 2)
        root.children[0].visit_count = 10
        root.children[0].value_sum = 5.0  # Q = 0.5
        root.children[1].visit_count = 1
        root.children[1].value_sum = 0.8  # Q = 0.8
        
        best_act, child = root.select(c_puct=1.0)
        self.assertIn(best_act, [0, 1])

    def test_mcts_search(self):
        model = DummyModel(board_size=15)
        mcts = MCTS(model, num_simulations=10, device="cpu")
        board = np.zeros((15, 15), dtype=np.int64)
        
        move, policy = mcts.search(board, active_player=1, step_count=0, temperature=1.0)
        self.assertEqual(len(policy), 225)
        self.assertEqual(policy.shape, (225,))
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertTrue(0 <= move[0] < 15)
        self.assertTrue(0 <= move[1] < 15)

if __name__ == "__main__":
    unittest.main()
