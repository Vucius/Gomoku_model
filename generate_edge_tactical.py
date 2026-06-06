import os
import pickle
import random
import numpy as np
from src.psq_parser import detect_threats

def generate_edge_tactical_dataset(output_path, num_samples=5000, board_size=15):
    """
    Generates synthetic Gomoku board states focusing on edge tactics.
    Saves the dataset as a list of dicts compatible with GomokuDataset:
    {
        "board": board (15x15 np.ndarray),
        "player": active player (1 or -1),
        "move": best response (r, c),
        "winner": winner result (0.0 or active player if winning),
        "step": step count (default 20),
        "source": "tactical"
    }
    """
    print(f"Generating {num_samples} edge tactical samples...")
    samples = []
    
    # Directions for alignment: horizontal, vertical, diagonal, anti-diagonal
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    while len(samples) < num_samples:
        board = np.zeros((board_size, board_size), dtype=np.int64)
        active_player = random.choice([1, -1])
        opponent = -active_player
        
        # 1. Select a target cell (r, c) near the edge (distance <= 2)
        r = random.randint(0, board_size - 1)
        c = random.randint(0, board_size - 1)
        dist = min(r, c, board_size - 1 - r, board_size - 1 - c)
        if dist > 2:
            # Re-sample to focus heavily on edges
            if random.random() > 0.1:  # 90% chance to force edge placement
                continue

        # 2. Decide threat type (Win/Four, Block, Live Three, Block Live Three)
        scenario = random.choice(["attack_four", "defend_four", "attack_three", "defend_three"])
        dr, dc = random.choice(directions)
        
        # We want to place stones in a line around (r, c)
        # e.g., for active player or opponent
        pattern_player = active_player if "attack" in scenario else opponent
        
        if "four" in scenario:
            # We want to construct a four-in-a-row threat (needs 4 stones total after playing at r, c)
            # Place 3 stones along the direction (dr, dc)
            offsets = [-3, -2, -1, 1, 2, 3]
            selected_offsets = random.sample(offsets, 3)
        else:
            # We want to construct a three-in-a-row threat (needs 3 stones total after playing at r, c)
            # Place 2 stones along the direction
            offsets = [-2, -1, 1, 2]
            selected_offsets = random.sample(offsets, 2)
            
        # Place stones
        valid = True
        placed_coords = []
        for offset in selected_offsets:
            nr, nc = r + dr * offset, c + dc * offset
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if board[nr][nc] == 0:
                    board[nr][nc] = pattern_player
                    placed_coords.append((nr, nc))
                else:
                    valid = False
            else:
                valid = False
                
        if not valid:
            continue
            
        # Add some random background noise (distractor stones far from the edge zone to prevent overfitting)
        num_distractors = random.randint(2, 6)
        distractors_added = 0
        while distractors_added < num_distractors:
            dr_rand = random.randint(0, board_size - 1)
            dc_rand = random.randint(0, board_size - 1)
            # Don't overwrite the active pattern or the target move
            if board[dr_rand][dc_rand] == 0 and (dr_rand, dc_rand) != (r, c) and (dr_rand, dc_rand) not in placed_coords:
                # Keep distractors mostly away from the immediate pattern line
                if abs(dr_rand - r) > 3 or abs(dc_rand - c) > 3:
                    board[dr_rand][dc_rand] = random.choice([1, -1])
                    distractors_added += 1
        
        # Verify the threat level of (r, c) using detect_threats
        threats_active = detect_threats(board.tolist(), active_player, board_size)
        threats_opponent = detect_threats(board.tolist(), opponent, board_size)
        
        active_threat = threats_active[r][c]
        opponent_threat = threats_opponent[r][c]
        
        # Decide if this move is indeed the best tactical option
        # For win/four attack: we want active threat >= 2
        # For defend/block four: we want opponent threat >= 2
        # For live three attack: active threat >= 1
        # For defend/block live three: opponent threat >= 1
        is_best_move = False
        if "attack" in scenario and active_threat >= 1:
            is_best_move = True
        elif "defend" in scenario and opponent_threat >= 1:
            # Blocking opponent threat at (r, c)
            is_best_move = True
            
        if is_best_move:
            # We found a valid tactical situation!
            samples.append({
                "board": board,
                "player": active_player,
                "move": (r, c),
                "winner": float(active_player) if active_threat >= 2 else 0.0,
                "step": random.randint(10, 40),
                "source": "tactical"
            })
            
    # Save the dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"Successfully generated and saved {len(samples)} tactical samples to {output_path}.")

if __name__ == "__main__":
    # Save to default folder under dataset/
    generate_edge_tactical_dataset("dataset/edge_tactical.pkl", num_samples=5000)
