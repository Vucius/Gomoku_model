import math
import numpy as np
import torch

class TreeNode:
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}  # action -> TreeNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select(self, c_puct=1.5):
        """
        Selects the action with the maximum PUCT value: Q(s, a) + U(s, a).
        """
        best_action = None
        best_value = -float("inf")
        
        sum_visits = sum(child.visit_count for child in self.children.values())
        sqrt_sum_visits = math.sqrt(sum_visits + 1e-8)
        
        for action, child in self.children.items():
            # PUCT formula
            u = c_puct * child.prior_prob * sqrt_sum_visits / (1 + child.visit_count)
            # Q-value is relative to this node's player (which is the opposite of the child's perspective)
            puct_val = -child.value + u
            
            if puct_val > best_value:
                best_value = puct_val
                best_action = action
                
        return best_action, self.children[best_action]

    def expand(self, action_probs):
        """
        Expands the node by creating new children with prior probabilities.
        action_probs: dict of action (int index) -> probability (float)
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = TreeNode(parent=self, prior_prob=prob)

    def update(self, value):
        """
        Updates the visit count and value sum.
        """
        self.visit_count += 1
        self.value_sum += value

    def backpropagate(self, value):
        """
        Backpropagates value up to the root.
        For alternative players, value flips sign: current player's value = -opponent's value.
        """
        node = self
        while node is not None:
            node.update(value)
            # Invert value for the parent player's perspective
            value = -value
            node = node.parent

class MCTS:
    def __init__(self, model, c_puct=1.5, num_simulations=400, device="cpu"):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        
    def search(self, board, active_player, step_count, temperature=1.0):
        """
        Performs MCTS search from the current board state.
        board: 15x15 numpy array, values in {0, 1, -1}
        active_player: 1 or -1
        step_count: current move count in game
        Returns:
            move: (r, c) tuple of the selected action
            policy: 225-dim probability distribution over the board
        """
        from src.dataset import build_features
        board_size = board.shape[0]
        root = TreeNode()
        
        # 1. Expand the root node first
        features_np = build_features(board, active_player, step_count, board_size)
        features_t = torch.from_numpy(features_np).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_t)
            policy_logits = outputs["policy_logits"].cpu().numpy()[0]
            
        # Get legal moves
        legal_mask = (board == 0).flatten()
        policy_exp = np.exp(policy_logits - np.max(policy_logits))
        policy_exp = policy_exp * legal_mask
        if policy_exp.sum() <= 0:
            # Fallback to uniform legal moves
            policy_probs = legal_mask / legal_mask.sum()
        else:
            policy_probs = policy_exp / policy_exp.sum()
            
        action_probs = {act: float(prob) for act, prob in enumerate(policy_probs) if legal_mask[act]}
        root.expand(action_probs)
        
        # 2. MCTS simulations loop
        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()
            current_player = active_player
            current_step = step_count
            
            # Selection
            path = [node]
            while len(node.children) > 0:
                action, child_node = node.select(self.c_puct)
                # Apply action to scratch board
                r_act, c_act = action // board_size, action % board_size
                scratch_board[r_act, c_act] = current_player
                # Alternate player
                current_player = -current_player
                current_step += 1
                node = child_node
                path.append(node)
                
            # Expansion and evaluation of leaf node
            # Check if this leaf node has a terminal state (5 in a row)
            from src.psq_parser import check_five_in_a_row
            winner = check_five_in_a_row(scratch_board, board_size)
            
            if winner != 0:
                # Value is +1.0 if the active player wins, -1.0 if they lose.
                # Since winner matches the player who just moved (-current_player),
                # the active player at the leaf (current_player) has lost, so value is -1.0.
                value = -1.0 if winner == -current_player else 1.0
            elif np.count_nonzero(scratch_board) == board_size * board_size:
                # Draw
                value = 0.0
            else:
                # Expand and evaluate with NN
                features_np = build_features(scratch_board, current_player, current_step, board_size)
                features_t = torch.from_numpy(features_np).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(features_t)
                    pol_logits = outputs["policy_logits"].cpu().numpy()[0]
                    value = float(outputs["value"].cpu().numpy()[0][0])
                    
                legal_mask = (scratch_board == 0).flatten()
                pol_exp = np.exp(pol_logits - np.max(pol_logits))
                pol_exp = pol_exp * legal_mask
                if pol_exp.sum() <= 0:
                    pol_probs = legal_mask / legal_mask.sum()
                else:
                    pol_probs = pol_exp / pol_exp.sum()
                    
                action_probs = {act: float(prob) for act, prob in enumerate(pol_probs) if legal_mask[act]}
                node.expand(action_probs)
                
            # Backpropagation
            # Note: the value returned by the NN is from current_player's perspective.
            # But the leaf node corresponds to the transition *to* current_player,
            # which is selected by the parent node.
            # So the backpropagated value should match the perspective of current_player
            node.backpropagate(value)
            
        # 3. Compute the search policy based on root visit counts
        visit_counts = np.zeros(board_size * board_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
            
        if temperature < 0.1:
            # Deterministic selection: argmax
            best_action = np.argmax(visit_counts)
            policy = np.zeros_like(visit_counts)
            policy[best_action] = 1.0
        else:
            # Softmax selection with temperature
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            sum_visits = visit_counts_temp.sum()
            if sum_visits <= 0:
                # Fallback
                policy = np.zeros_like(visit_counts)
                policy[np.argmax(visit_counts)] = 1.0
            else:
                policy = visit_counts_temp / sum_visits
                
        # Sample move from the policy
        action = np.random.choice(len(policy), p=policy)
        move = (action // board_size, action % board_size)
        
        return move, policy
