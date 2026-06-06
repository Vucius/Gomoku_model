import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler

from src.config import GomokuConfig
from src.psq_parser import parse_psq_file, detect_threats
from src.augmentation import apply_d4_symmetry

_edge_mask_cache = {}
def get_edge_mask(board_size=15):
    if board_size not in _edge_mask_cache:
        mask = np.zeros((board_size, board_size), dtype=np.int64)
        for r in range(board_size):
            for c in range(board_size):
                dist = min(r, c, board_size - 1 - r, board_size - 1 - c)
                if dist <= 2:
                    mask[r, c] = 1
        _edge_mask_cache[board_size] = mask
    return _edge_mask_cache[board_size]

_edge_distance_cache = {}
def get_edge_distance_matrix(board_size=15):
    """
    Computes a board_size x board_size matrix containing normalized distance to the nearest edge.
    """
    if board_size not in _edge_distance_cache:
        matrix = np.zeros((board_size, board_size), dtype=np.float32)
        max_dist = (board_size - 1) / 2.0
        for r in range(board_size):
            for c in range(board_size):
                matrix[r, c] = min(r, c, board_size - 1 - r, board_size - 1 - c) / max_dist
        _edge_distance_cache[board_size] = matrix
    return _edge_distance_cache[board_size]

_coord_conv_cache = {}
def get_coord_conv_matrices(board_size=15):
    """
    Computes X and Y coordinate grids normalized to [-1, 1].
    """
    if board_size not in _coord_conv_cache:
        x_coords = np.linspace(-1, 1, board_size, dtype=np.float32)
        y_coords = np.linspace(-1, 1, board_size, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        _coord_conv_cache[board_size] = (xx, yy)
    return _coord_conv_cache[board_size]

_center_distance_cache = {}
def get_center_distance_matrix(board_size=15):
    """
    Computes Euclidean distance to center (board_size/2, board_size/2) normalized to [0, 1].
    """
    if board_size not in _center_distance_cache:
        center = (board_size - 1) / 2.0
        matrix = np.zeros((board_size, board_size), dtype=np.float32)
        max_dist = np.sqrt(2 * (center ** 2))
        for r in range(board_size):
            for c in range(board_size):
                matrix[r, c] = np.sqrt((r - center)**2 + (c - center)**2) / max_dist
        _center_distance_cache[board_size] = matrix
    return _center_distance_cache[board_size]

def build_features(board_state, active_player, step_count, board_size=15):
    """
    Builds the 8 feature channels from raw board state and active player perspective.
    Args:
        board_state: numpy array of shape (H, W), values in {0, 1, -1}
        active_player: int, 1 or -1
        step_count: int, step number in the game
    Returns:
        features: numpy array of shape (8, H, W)
    """
    # 1. Perspective check: convert to active player perspective
    # active player's stones = 1, opponent's = -1
    board_rel = board_state * active_player
    
    c0 = (board_rel == 1).astype(np.float32)    # Active player stones
    c1 = (board_rel == -1).astype(np.float32)   # Opponent stones
    c2 = (board_rel == 0).astype(np.float32)    # Legal moves

    # 4. Normalized edge distance
    c3 = get_edge_distance_matrix(board_size)

    # 5 & 6. CoordConv X, Y
    c4, c5 = get_coord_conv_matrices(board_size)

    # 7. Center distance
    c6 = get_center_distance_matrix(board_size)

    # 8. Step count / stage plane
    c7 = np.full((board_size, board_size), step_count / float(board_size * board_size), dtype=np.float32)

    return np.stack([c0, c1, c2, c3, c4, c5, c6, c7], axis=0)

class GomokuDataset(Dataset):
    """
    Multi-source Gomoku Dataset combining parsed .psq files and .npy pre-split arrays.
    """
    def __init__(self, config, is_train=True, use_psq=True, use_npy=True):
        self.config = config
        self.board_size = config.BOARD_SIZE
        self.is_train = is_train
        
        self.samples = []
        
        # 1. Load from pre-split numpy arrays (winepy dataset)
        if use_npy and os.path.exists(config.WINEPY_SPLIT_DIR):
            split_subdir = "train" if is_train else "test"
            full_board_path = os.path.join(config.WINEPY_SPLIT_DIR, split_subdir, "full_board")
            
            states_file = os.path.join(full_board_path, "board_states.npy")
            coords_file = os.path.join(full_board_path, "next_moves_coords.npy")
            players_file = os.path.join(full_board_path, "next_moves_players.npy")
            
            if os.path.exists(states_file) and os.path.exists(coords_file):
                states = np.load(states_file)
                coords = np.load(coords_file)
                players = np.load(players_file) if os.path.exists(players_file) else np.ones(len(states))
                
                # We assume the winner target is unknown for individual states, default value = 0 (draw)
                for idx in range(len(states)):
                    board = states[idx]
                    player = players[idx]
                    # Numpy split dataset coords are stored as (x, y), convert to (row=y, col=x)
                    x, y = coords[idx]
                    row, col = int(y), int(x)
                    
                    # Store
                    self.samples.append({
                        "board": board,
                        "player": int(player),
                        "move": (row, col),
                        "winner": 0.0,
                        "step": 20,  # Proxy average step count
                        "source": "npy"
                    })

        # 2. Load and parse .psq tournament files (gomocup2017)
        if use_psq and os.path.exists(config.GOMOCUP_DIR):
            cache_file = os.path.join(config.BASE_DIR, f"psq_cache_{'train' if is_train else 'test'}.pkl")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    psq_samples = pickle.load(f)
                self.samples.extend(psq_samples)
            else:
                psq_samples = []
                # Gather all .psq files recursively
                psq_files = glob.glob(os.path.join(config.GOMOCUP_DIR, "**", "*.psq"), recursive=True)
                
                # Split at game level: 80% train, 20% test
                np.random.seed(42)
                np.random.shuffle(psq_files)
                split_idx = int(len(psq_files) * 0.8)
                
                selected_files = psq_files[:split_idx] if is_train else psq_files[split_idx:]
                
                for filepath in selected_files:
                    parsed = parse_psq_file(filepath, self.board_size)
                    if not parsed or len(parsed["states_seq"]) == 0:
                        continue
                    
                    winner = parsed["winner"]
                    for step_info in parsed["states_seq"]:
                        psq_samples.append({
                            "board": np.array(step_info["board"]),
                            "player": step_info["player"],
                            "move": step_info["move"],
                            "winner": float(winner),
                            "step": step_info["step"],
                            "source": "psq"
                        })
                
                # Cache parsed results
                with open(cache_file, 'wb') as f:
                    pickle.dump(psq_samples, f)
                self.samples.extend(psq_samples)

        # 3. Load edge tactical generated dataset (Stage 4)
        tactical_file = os.path.join(config.BASE_DIR, "dataset", "edge_tactical.pkl")
        if os.path.exists(tactical_file):
            print(f"Loading edge tactical dataset from {tactical_file}...")
            with open(tactical_file, "rb") as f:
                tactical_samples = pickle.load(f)
            
            # Split: 80% train, 20% validation
            random_state = np.random.RandomState(42)
            indices = np.arange(len(tactical_samples))
            random_state.shuffle(indices)
            split_idx = int(len(indices) * 0.8)
            
            selected_indices = indices[:split_idx] if is_train else indices[split_idx:]
            for idx in selected_indices:
                self.samples.append(tactical_samples[idx])
            print(f"Added {len(selected_indices)} edge tactical samples to {'train' if is_train else 'val'} dataset.")

        # 4. Load self-play generated dataset (Stage 5)
        self_play_dir = os.path.join(config.BASE_DIR, "dataset", "self_play_data")
        if os.path.exists(self_play_dir):
            self_play_files = glob.glob(os.path.join(self_play_dir, "*.pkl"))
            if len(self_play_files) > 0:
                print(f"Loading {len(self_play_files)} self-play game files...")
                loaded_games = []
                for filepath in self_play_files:
                    with open(filepath, "rb") as f:
                        game_samples = pickle.load(f)
                    loaded_games.extend(game_samples)
                
                # Split: 80% train, 20% validation
                random_state = np.random.RandomState(42)
                indices = np.arange(len(loaded_games))
                random_state.shuffle(indices)
                split_idx = int(len(indices) * 0.8)
                
                selected_indices = indices[:split_idx] if is_train else indices[split_idx:]
                for idx in selected_indices:
                    self.samples.append(loaded_games[idx])
                print(f"Added {len(selected_indices)} self-play samples to {'train' if is_train else 'val'} dataset.")

        # Categorize indices for edge/corner tactical resampling
        self.normal_indices = []
        self.edge_indices = []
        self.corner_indices = []

        for idx, sample in enumerate(self.samples):
            if sample.get("source") == "self_play":
                policy = sample["policy"]
                best_act = int(np.argmax(policy))
                r = best_act // self.board_size
                c = best_act % self.board_size
            else:
                r, c = sample["move"]
            min_dist = min(r, c, self.board_size - 1 - r, self.board_size - 1 - c)
            
            is_corner = (r <= 1 or r >= self.board_size - 2) and (c <= 1 or c >= self.board_size - 2)
            
            if min_dist <= 1:
                if is_corner:
                    self.corner_indices.append(idx)
                else:
                    self.edge_indices.append(idx)
            else:
                self.normal_indices.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        board = sample["board"]
        player = sample["player"]
        move = sample["move"]
        winner = sample["winner"]
        step = sample["step"]
        
        # Lazy cache the base threat grid
        if "threat_grid" not in sample:
            sample["threat_grid"] = np.array(detect_threats(board, player, self.board_size), dtype=np.int64)
        
        threat_grid = sample["threat_grid"]

        # Create policy target (distribution map)
        if sample.get("source") == "self_play":
            policy_target = sample["policy"].reshape(self.board_size, self.board_size).copy()
        else:
            policy_target = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            policy_target[move[0], move[1]] = 1.0

        # Apply D4 symmetry transformations during training (rotations + reflections)
        if self.is_train:
            sym_idx = np.random.randint(0, 8)
            board_3d = board[np.newaxis, :, :]
            sym_board_3d, policy_target = apply_d4_symmetry(board_3d, policy_target, sym_idx)
            board = sym_board_3d[0]
            
            # Apply D4 symmetry to threat_grid
            rot_k = sym_idx % 4
            flip_h = sym_idx // 4
            threat_grid_sym = threat_grid.copy()
            if flip_h:
                threat_grid_sym = np.flip(threat_grid_sym, axis=1)
            if rot_k > 0:
                threat_grid_sym = np.rot90(threat_grid_sym, k=rot_k, axes=(0, 1))
            threat_grid = threat_grid_sym
            
        # Build features
        features = build_features(board, player, step, self.board_size)
        
        # Create value target (outcome relative to active player)
        if sample.get("source") == "self_play":
            value_target = np.array([winner], dtype=np.float32)
        else:
            value_target = np.array([winner * player], dtype=np.float32)

        # Generate auxiliary targets
        # 1. Threat map (detect threats for current player on board before making the move)
        threat_target = np.array(threat_grid, dtype=np.int64)

        # 2. Edge Threat: Threat map masked to borders (distance <= 2 to edge) and binarized
        edge_threat_target = (threat_target * get_edge_mask(self.board_size) > 0).astype(np.float32)

        # 3. Distance-to-Win Target (Proxy estimation: number of remaining moves in game / max possible moves)
        # If no win occurred, target is 1.0 (far)
        if winner == 0:
            dtw_target = np.array([1.0], dtype=np.float32)
        else:
            # Estimate: steps remaining to win
            remaining_steps = max(0, 100 - step) # Heuristic remaining steps
            dtw_target = np.array([remaining_steps / 100.0], dtype=np.float32)

        # 4. Legal moves mask (empty intersections)
        legal_move_target = (board == 0).astype(np.float32)

        # Convert to Tensors (zero-copy using torch.from_numpy)
        return {
            "features": torch.from_numpy(features),
            "policy": torch.from_numpy(policy_target),
            "value": torch.from_numpy(value_target),
            "threat_target": torch.from_numpy(np.ascontiguousarray(threat_target)),
            "edge_threat_target": torch.from_numpy(edge_threat_target).unsqueeze(0),
            "dtw_target": torch.from_numpy(dtw_target),
            "legal_move_target": torch.from_numpy(legal_move_target).unsqueeze(0)
        }

class TacticalEdgeBatchSampler(BatchSampler):
    """
    Batch Sampler that structures batches with specific ratios:
    - 40% normal moves
    - 20% edge moves
    - 20% edge threats (or normal if not enough)
    - 10% corner moves
    - 10% normal/other moves
    This solves the edge insensitivity issue by oversampling boundary tactics.
    """
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Calculate samples per batch
        # Using config: 40% normal, 20% edge_moves, 20% edge_threats, 10% corner_moves, 10% tactical
        self.num_normal = int(batch_size * 0.50)  # Combining normal + tactical (50%)
        self.num_edge = int(batch_size * 0.40)    # Combining edge_moves + edge_threats (40%)
        self.num_corner = batch_size - self.num_normal - self.num_edge # Corner (10%)

    def __iter__(self):
        # Shuffle indices
        normal_idx = np.random.permutation(self.dataset.normal_indices)
        edge_idx = np.random.permutation(self.dataset.edge_indices)
        corner_idx = np.random.permutation(self.dataset.corner_indices)

        # Iterators
        normal_ptr = 0
        edge_ptr = 0
        corner_ptr = 0

        # Calculate number of batches
        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            batch = []
            
            # 1. Normal samples
            for _ in range(self.num_normal):
                if normal_ptr >= len(normal_idx):
                    normal_idx = np.random.permutation(self.dataset.normal_indices)
                    normal_ptr = 0
                if len(normal_idx) > 0:
                    batch.append(int(normal_idx[normal_ptr]))
                    normal_ptr += 1

            # 2. Edge samples
            for _ in range(self.num_edge):
                if edge_ptr >= len(edge_idx):
                    edge_idx = np.random.permutation(self.dataset.edge_indices)
                    edge_ptr = 0
                if len(edge_idx) > 0:
                    batch.append(int(edge_idx[edge_ptr]))
                    edge_ptr += 1

            # 3. Corner samples
            for _ in range(self.num_corner):
                if corner_ptr >= len(corner_idx):
                    corner_idx = np.random.permutation(self.dataset.corner_indices)
                    corner_ptr = 0
                if len(corner_idx) > 0:
                    batch.append(int(corner_idx[corner_ptr]))
                    corner_ptr += 1

            # If the batch is not full (e.g. some lists are empty), backfill with normal
            while len(batch) < self.batch_size:
                if normal_ptr >= len(normal_idx):
                    normal_idx = np.random.permutation(self.dataset.normal_indices)
                    normal_ptr = 0
                if len(normal_idx) > 0:
                    batch.append(int(normal_idx[normal_ptr]))
                    normal_ptr += 1
                else:
                    break

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
