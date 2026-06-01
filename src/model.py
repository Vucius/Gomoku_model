import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbedding(nn.Module):
    """
    Learnable 2D Position Embedding module.
    Can either use a full (C x H x W) learnable tensor or decompose it
    into row-wise and column-wise learnable parameters to reduce parameters.
    """
    def __init__(self, channels, height, width, decompose=False):
        super().__init__()
        self.decompose = decompose
        if decompose:
            self.row_embed = nn.Parameter(torch.randn(1, channels, height, 1) * 0.02)
            self.col_embed = nn.Parameter(torch.randn(1, channels, 1, width) * 0.02)
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)

    def forward(self, x):
        # x shape: (B, C, H, W)
        if self.decompose:
            return x + self.row_embed + self.col_embed
        else:
            return x + self.pos_embed

class ResBlock(nn.Module):
    """
    Standard Residual Block with Batch Normalization and ReLU activation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class GomokuNet(nn.Module):
    """
    Gomoku neural network with multiple policy, value, and auxiliary heads.
    Designed to prevent edge insensitivity and policy collapse.
    """
    def __init__(self, config):
        super().__init__()
        self.board_size = config.BOARD_SIZE
        self.in_channels = config.NUM_INPUT_CHANNELS
        self.hidden_channels = config.HIDDEN_CHANNELS
        self.num_res_blocks = config.NUM_RES_BLOCKS
        self.num_threat_classes = config.NUM_THREAT_CLASSES

        # Initial Convolution
        self.conv_init = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(self.hidden_channels)

        # Position Embedding
        if config.USE_LEARNABLE_POS_EMBED:
            self.pos_embedding = PositionEmbedding(self.hidden_channels, self.board_size, self.board_size)
        else:
            self.pos_embedding = None

        # Residual Backbone
        self.backbone = nn.ModuleList([ResBlock(self.hidden_channels) for _ in range(self.num_res_blocks)])

        # --- 1. Policy Head ---
        self.policy_conv = nn.Conv2d(self.hidden_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_size * self.board_size, self.board_size * self.board_size)

        # --- 2. Value Head ---
        self.value_conv = nn.Conv2d(self.hidden_channels, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * self.board_size * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # --- 3. Auxiliary Heads (For Multi-Task Training) ---
        
        # Threat Type Head: Segment threat categories (e.g. none, live-3, rush-4, live-4, etc.)
        self.threat_type_conv = nn.Conv2d(self.hidden_channels, self.num_threat_classes, kernel_size=3, padding=1)

        # Edge Threat Head: Binary segmentation focus near edges (within 2 steps of boundary)
        self.edge_threat_conv = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, padding=1)

        # Distance-to-Win Head: Regress number of steps until win/loss (scalar output)
        self.dist_to_win_conv = nn.Conv2d(self.hidden_channels, 8, kernel_size=1, bias=False)
        self.dist_to_win_bn = nn.BatchNorm2d(8)
        self.dist_to_win_fc1 = nn.Linear(8 * self.board_size * self.board_size, 128)
        self.dist_to_win_fc2 = nn.Linear(128, 1)

        # Legal Move Head: Auxiliary segmentation for empty/legal intersections
        self.legal_move_conv = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Input x shape: (B, NUM_INPUT_CHANNELS, H, W)
        
        # Initial features
        out = F.relu(self.bn_init(self.conv_init(x)))
        
        # Apply Position Embedding
        if self.pos_embedding is not None:
            out = self.pos_embedding(out)

        # ResNet trunk
        for res_block in self.backbone:
            out = res_block(out)

        # 1. Policy Head computation
        pol = F.relu(self.policy_bn(self.policy_conv(out)))
        pol = pol.view(pol.size(0), -1)
        policy_logits = self.policy_fc(pol)  # Output shape: (B, H*W)

        # 2. Value Head computation
        val = F.relu(self.value_bn(self.value_conv(out)))
        val = val.view(val.size(0), -1)
        val = F.relu(self.value_fc1(val))
        value = torch.tanh(self.value_fc2(val))  # Output shape: (B, 1)

        # 3. Auxiliary Outputs (training only)
        # Threat Type Head: outputs segmentation logits (B, NUM_THREAT_CLASSES, H, W)
        threat_logits = self.threat_type_conv(out)

        # Edge Threat Head: outputs threat map near edges (B, 1, H, W)
        edge_threat_logits = self.edge_threat_conv(out)

        # Distance-to-Win Head: predicts number of remaining steps (B, 1)
        dtw = F.relu(self.dist_to_win_bn(self.dist_to_win_conv(out)))
        dtw = dtw.view(dtw.size(0), -1)
        dtw = F.relu(self.dist_to_win_fc1(dtw))
        distance_to_win = self.dist_to_win_fc2(dtw)

        # Legal Move Head: predicts board occupancy map (B, 1, H, W)
        legal_move_logits = self.legal_move_conv(out)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "threat_logits": threat_logits,
            "edge_threat_logits": edge_threat_logits,
            "distance_to_win": distance_to_win,
            "legal_move_logits": legal_move_logits
        }
