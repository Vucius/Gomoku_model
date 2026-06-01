import os

class GomokuConfig:
    # --- Board Settings ---
    BOARD_SIZE = 15
    WIN_COUNT = 5  # Standard Gomoku is 5 in a row

    # --- Dataset & Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GOMOCUP_DIR = os.path.join(BASE_DIR, "dataset", "gomocup2017")
    WINEPY_SPLIT_DIR = os.path.join(BASE_DIR, "dataset", "gomoku_dataset_split")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    # --- Feature Channels ---
    # 1: current player, 2: opponent player, 3: legal moves,
    # 4: normalized edge distance, 5: coord_conv_x, 6: coord_conv_y,
    # 7: center distance (optional), 8: step count (optional)
    NUM_INPUT_CHANNELS = 8

    # --- Model Architecture ---
    HIDDEN_CHANNELS = 128
    NUM_RES_BLOCKS = 12
    AUX_HEAD_LOSS_WEIGHT = 0.1

    # --- Position Embedding Settings ---
    USE_LEARNABLE_POS_EMBED = True
    POS_EMBED_DIM = 128  # Matches HIDDEN_CHANNELS

    # --- Loss Weights & Parameters ---
    POLICY_ENTROPY_ALPHA = 0.01  # Policy entropy regularization weight
    LABEL_SMOOTHING_EPS = 0.05  # Policy label smoothing epsilon (Stage 3)
    TEMPERATURE = 1.0  # Softmax temperature (Stage 3)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda"  # Will fallback to "cpu" if CUDA is unavailable in trainer

    # --- Teacher Distillation Config (Stage 3) ---
    TEACHER_POLICY_WEIGHT = 0.75  # Weight of teacher model soft targets
    TEACHER_VALUE_WEIGHT = 0.75

    # --- Edge Tactical Resampling Config (Stage 4) ---
    # Targets for the batch composition (Stage 4)
    BATCH_COMPOSITION = {
        "normal": 0.40,        # Normal self-play / human matches
        "edge_moves": 0.20,    # Move is within distance 1 of boundary
        "edge_threats": 0.20,  # Threat is near edge
        "corner_moves": 0.10,  # Move is within distance 1 of corner
        "tactical": 0.10       # Artificial pattern matches
    }

    # --- Threat categories for auxiliary threat head (Stage 1.3) ---
    THREAT_CLASSES = {
        0: "none",
        1: "live_three",  # 活三
        2: "rush_four",   # 冲四
        3: "live_four",   # 活四
        4: "double_three",# 双三
        5: "four_three"   # 四三
    }
    NUM_THREAT_CLASSES = 6
