import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuMultiTaskLoss(nn.Module):
    """
    Custom Multi-Task Loss for Gomoku.
    Combines Policy Loss (with label smoothing and entropy regularization),
    Value Loss (MSE), and auxiliary heads losses.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha = config.POLICY_ENTROPY_ALPHA
        self.epsilon = config.LABEL_SMOOTHING_EPS
        self.aux_weight = config.AUX_HEAD_LOSS_WEIGHT

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: dict from GomokuNet containing:
                "policy_logits": (B, H*W)
                "value": (B, 1)
                "threat_logits": (B, C_threat, H, W)
                "edge_threat_logits": (B, 1, H, W)
                "distance_to_win": (B, 1)
                "legal_move_logits": (B, 1, H, W)
            targets: dict containing:
                "policy": (B, H, W) or (B, H*W)
                "value": (B, 1)
                "threat_target": (B, H, W)
                "edge_threat_target": (B, 1, H, W)
                "dtw_target": (B, 1)
                "legal_move_target": (B, 1, H, W)
        """
        # --- 1. Policy Loss with Label Smoothing & Entropy Regularization ---
        policy_logits = model_outputs["policy_logits"]
        log_policy_probs = F.log_softmax(policy_logits, dim=-1)
        
        target_policy = targets["policy"].view(targets["policy"].size(0), -1)
        legal_mask = targets["legal_move_target"].view(targets["legal_move_target"].size(0), -1)
        legal_mask_bool = legal_mask.bool()
        
        # Policy Label Smoothing
        # Smoothes one-hot targets across all legal moves to prevent extreme policy predictions
        num_legal = legal_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform_legal = legal_mask / num_legal
        smoothed_target = (1.0 - self.epsilon) * target_policy + self.epsilon * uniform_legal
        
        # Cross Entropy Loss
        policy_loss = -torch.mean(torch.sum(smoothed_target * log_policy_probs, dim=1))
        
        # Policy Entropy Regularization over legal moves only.
        # This prevents target collapse without rewarding probability mass on occupied points.
        legal_logits = policy_logits.masked_fill(~legal_mask_bool, torch.finfo(policy_logits.dtype).min)
        legal_probs = F.softmax(legal_logits, dim=-1)
        legal_log_probs = F.log_softmax(legal_logits, dim=-1)
        entropy = -torch.sum(legal_probs * legal_log_probs.masked_fill(~legal_mask_bool, 0.0), dim=1)
        entropy_loss = -torch.mean(entropy)  # Negative entropy to maximize it
        
        total_policy_loss = policy_loss + self.alpha * entropy_loss

        # --- 2. Value Loss ---
        value_pred = model_outputs["value"]
        value_target = targets["value"]
        value_loss = F.mse_loss(value_pred, value_target)

        # --- 3. Auxiliary Losses ---
        
        # Threat Type segmentation (multi-class cross entropy)
        threat_logits = model_outputs["threat_logits"]
        threat_target = targets["threat_target"]
        threat_loss = F.cross_entropy(threat_logits, threat_target)

        # Edge Threat segmentation (binary cross entropy)
        edge_threat_logits = model_outputs["edge_threat_logits"]
        edge_threat_target = targets["edge_threat_target"]
        edge_threat_loss = F.binary_cross_entropy_with_logits(edge_threat_logits, edge_threat_target)

        # Distance-to-win regression (MSE)
        dtw_pred = model_outputs["distance_to_win"]
        dtw_target = targets["dtw_target"]
        dtw_loss = F.mse_loss(dtw_pred, dtw_target)

        # Legal move segmentation (binary cross entropy)
        legal_move_logits = model_outputs["legal_move_logits"]
        legal_move_target = targets["legal_move_target"]
        legal_move_loss = F.binary_cross_entropy_with_logits(legal_move_logits, legal_move_target)

        # Sum of auxiliary losses
        aux_loss = threat_loss + edge_threat_loss + dtw_loss + legal_move_loss

        # --- Total Combined Multi-Task Loss ---
        total_loss = total_policy_loss + value_loss + self.aux_weight * aux_loss

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "entropy": torch.mean(entropy),
            "value_loss": value_loss,
            "aux_loss": aux_loss,
            "threat_loss": threat_loss,
            "edge_threat_loss": edge_threat_loss,
            "dtw_loss": dtw_loss,
            "legal_move_loss": legal_move_loss
        }
