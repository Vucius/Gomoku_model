import os
import glob
import time
import random
import pickle
import argparse
import numpy as np
import torch

from src.config import GomokuConfig
from src.model import GomokuNet
from src.mcts import MCTS
from src.psq_parser import check_five_in_a_row

def get_checkpoints(checkpoint_dir):
    """
    Returns a sorted list of absolute checkpoint paths in checkpoint_dir.
    """
    if not os.path.exists(checkpoint_dir):
        return []
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    # Filter out backing files that aren't model weights if any
    return sorted(pt_files)

def load_model_from_checkpoint(checkpoint_path, default_config, device):
    """
    Reconstructs and loads a model from checkpoint path, extracting its configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    import copy
    model_config = copy.deepcopy(default_config)
    if "config" in checkpoint:
        loaded_conf = checkpoint["config"]
        if hasattr(loaded_conf, "NUM_RES_BLOCKS"):
            model_config.NUM_RES_BLOCKS = loaded_conf.NUM_RES_BLOCKS
        if hasattr(loaded_conf, "HIDDEN_CHANNELS"):
            model_config.HIDDEN_CHANNELS = loaded_conf.HIDDEN_CHANNELS
            
    model = GomokuNet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def play_game(black_model, white_model, simulations, temp_threshold, device, board_size=15):
    """
    Simulates a single self-play game using two models.
    """
    board = np.zeros((board_size, board_size), dtype=np.int64)
    black_mcts = MCTS(black_model, num_simulations=simulations, device=device)
    white_mcts = MCTS(white_model, num_simulations=simulations, device=device)
    
    game_history = []
    step_count = 0
    current_player = 1  # Black starts (represented by 1)
    
    while True:
        # Temperature schedule: higher temp at start for exploration
        temperature = 1.0 if step_count < temp_threshold else 0.01
        
        # MCTS search
        mcts = black_mcts if current_player == 1 else white_mcts
        move, search_policy = mcts.search(board, current_player, step_count, temperature)
        
        # Store transition data
        game_history.append({
            "board": board.copy(),
            "player": current_player,
            "policy": search_policy,
            "step": step_count
        })
        
        # Apply move
        board[move[0], move[1]] = current_player
        step_count += 1
        
        # Check game end
        winner = check_five_in_a_row(board, board_size)
        if winner != 0:
            print(f"Game finished in {step_count} moves. Winner: {winner}")
            break
        elif np.count_nonzero(board) == board_size * board_size:
            print(f"Game finished in {step_count} moves. Draw.")
            winner = 0
            break
            
        # Switch player
        current_player = -current_player
        
    # Build training samples with value relative to each active player
    samples = []
    for state in game_history:
        player = state["player"]
        # Value target: +1.0 if player wins, -1.0 if opponent wins, 0.0 for draw
        if winner == 0:
            val_target = 0.0
        else:
            val_target = 1.0 if winner == player else -1.0
            
        samples.append({
            "board": state["board"],
            "player": player,
            "move": None,  # Not used; policy targets are extracted from search policy distribution
            "winner": val_target,  # We store value target here directly
            "step": state["step"],
            "policy": state["policy"],  # Entire search distribution
            "source": "self_play"
        })
        
    return samples

def main():
    parser = argparse.ArgumentParser(description="Gomoku MCTS Self-Play Data Generator")
    parser.add_argument("--num_games", type=int, default=10, help="Number of self-play games to simulate")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations count per step")
    parser.add_argument("--temp_threshold", type=int, default=15, help="Number of early moves to use tau=1.0")
    parser.add_argument("--output_dir", type=str, default="dataset/self_play_data", help="Replay buffer save path")
    args = parser.parse_args()
    
    config = GomokuConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Self-play initialized on device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints = get_checkpoints(config.CHECKPOINT_DIR)
    
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {config.CHECKPOINT_DIR}. Instantiating raw untrained models.")
        latest_model = GomokuNet(config).to(device)
        best_model = latest_model
        historical_checkpoints = []
    else:
        print(f"Found {len(checkpoints)} checkpoints. Setting up Model Pool...")
        latest_path = checkpoints[-1]
        print(f"Latest model: {latest_path}")
        latest_model = load_model_from_checkpoint(latest_path, config, device)
        
        # Try to find a 'best_*.pt' or 'base_model.pt' for historical comparison
        best_candidates = [c for c in checkpoints if "best" in c or "base_model.pt" in c]
        best_path = best_candidates[-1] if len(best_candidates) > 0 else checkpoints[-1]
        print(f"Best model candidate: {best_path}")
        best_model = load_model_from_checkpoint(best_path, config, device)
        
        historical_checkpoints = checkpoints[:-1]

    for game_idx in range(1, args.num_games + 1):
        print(f"\n--- Simulating Game {game_idx}/{args.num_games} ---")
        
        # Decide matchup from Model Pool
        # 60% latest vs latest, 20% latest vs best, 20% latest vs older
        r = random.random()
        if len(historical_checkpoints) == 0:
            black_model, white_model = latest_model, latest_model
            p1_name, p2_name = "Latest", "Latest"
        else:
            if r < 0.6:
                black_model, white_model = latest_model, latest_model
                p1_name, p2_name = "Latest", "Latest"
            elif r < 0.8:
                # Black = Latest, White = Best (or vice-versa)
                if random.choice([True, False]):
                    black_model, white_model = latest_model, best_model
                    p1_name, p2_name = "Latest", "Best"
                else:
                    black_model, white_model = best_model, latest_model
                    p1_name, p2_name = "Best", "Latest"
            else:
                # Black = Latest, White = Older random checkpoint
                old_path = random.choice(historical_checkpoints)
                old_model = load_model_from_checkpoint(old_path, config, device)
                if random.choice([True, False]):
                    black_model, white_model = latest_model, old_model
                    p1_name, p2_name = "Latest", os.path.basename(old_path)
                else:
                    black_model, white_model = old_model, latest_model
                    p1_name, p2_name = os.path.basename(old_path), "Latest"
                    
        print(f"Matchup: Black ({p1_name}) vs White ({p2_name})")
        
        t0 = time.time()
        samples = play_game(
            black_model, 
            white_model, 
            args.simulations, 
            args.temp_threshold, 
            device, 
            config.BOARD_SIZE
        )
        duration = time.time() - t0
        print(f"Simulated game in {duration:.1f} seconds. Generated {len(samples)} samples.")
        
        # Save game data
        timestamp = int(time.time() * 1000)
        output_file = os.path.join(args.output_dir, f"game_{timestamp}_{game_idx}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(samples, f)
            
    print(f"\nAll {args.num_games} self-play games generated and saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
