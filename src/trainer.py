import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loss import GomokuMultiTaskLoss

class GomokuTrainer:
    """
    Manager for training and evaluating GomokuNet.
    Includes support for knowledge distillation, temperature scaling, and checkpointing.
    """
    def __init__(self, config, model, train_loader, val_loader, teacher_model=None):
        self.config = config
        if config.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.teacher_model = teacher_model.to(self.device) if teacher_model is not None else None

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        self.criterion = GomokuMultiTaskLoss(config)

        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)

        # Dynamic training hyperparameters
        self.temperature = config.TEMPERATURE
        self.teacher_policy_weight = config.TEACHER_POLICY_WEIGHT
        self.teacher_value_weight = config.TEACHER_VALUE_WEIGHT
        self.policy_top_k = config.POLICY_TOP_K
        self.policy_top_k_floor = config.POLICY_TOP_K_FLOOR
        self.enable_top_k_policy = config.ENABLE_TOP_K_POLICY

    def train_epoch(self, epoch):
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_aux_loss = 0.0
        correct_moves = 0
        total_samples = 0

        device_type = "xpu" if "xpu" in str(self.device) else ("cuda" if "cuda" in str(self.device) else "cpu")
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Train]")
        for batch in loop:
            # Move data to device
            features = batch["features"].to(self.device, non_blocking=True)
            target_policy = batch["policy"].to(self.device, non_blocking=True)
            target_value = batch["value"].to(self.device, non_blocking=True)
            threat_target = batch["threat_target"].to(self.device, non_blocking=True)
            edge_threat_target = batch["edge_threat_target"].to(self.device, non_blocking=True)
            dtw_target = batch["dtw_target"].to(self.device, non_blocking=True)
            legal_move_target = batch["legal_move_target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Knowledge Distillation (Stage 3)
                # If teacher model is available, mix standard targets with teacher predictions
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(features)
                        teacher_policy_soft = torch.softmax(teacher_outputs["policy_logits"], dim=-1)
                        teacher_value_soft = teacher_outputs["value"]
                    
                    # Reshape soft policy to match targets
                    target_policy_flat = target_policy.view(target_policy.size(0), -1)
                    mixed_policy = (1.0 - self.teacher_policy_weight) * target_policy_flat + self.teacher_policy_weight * teacher_policy_soft
                    target_policy = mixed_policy.view_as(target_policy)

                    mixed_value = (1.0 - self.teacher_value_weight) * target_value + self.teacher_value_weight * teacher_value_soft
                    target_value = mixed_value

                target_policy = self.shape_policy_target(target_policy, legal_move_target)

                targets = {
                    "policy": target_policy,
                    "value": target_value,
                    "threat_target": threat_target,
                    "edge_threat_target": edge_threat_target,
                    "dtw_target": dtw_target,
                    "legal_move_target": legal_move_target
                }
            
                # Forward pass
                outputs = self.model(features)
                
                # Apply Temperature scaling to policy logits to control policy smoothing
                if self.temperature != 1.0:
                    outputs["policy_logits"] = outputs["policy_logits"] / self.temperature

                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["loss"]

            loss.backward()
            self.optimizer.step()

            # Track metrics
            batch_size = features.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_policy_loss += loss_dict["policy_loss"].item() * batch_size
            epoch_value_loss += loss_dict["value_loss"].item() * batch_size
            epoch_aux_loss += loss_dict["aux_loss"].item() * batch_size

            # Move accuracy (argmax index matching original target move location)
            pred_moves = outputs["policy_logits"].argmax(dim=-1)
            # Find the actual ground truth move (before distillation mixing)
            gt_moves = batch["policy"].view(batch_size, -1).argmax(dim=-1).to(self.device)
            correct_moves += (pred_moves == gt_moves).sum().item()
            total_samples += batch_size

            loop.set_postfix({
                "loss": loss.item(),
                "pol_l": loss_dict["policy_loss"].item(),
                "val_l": loss_dict["value_loss"].item(),
                "acc": (pred_moves == gt_moves).float().mean().item()
            })

        avg_loss = epoch_loss / total_samples
        avg_pol_loss = epoch_policy_loss / total_samples
        avg_val_loss = epoch_value_loss / total_samples
        avg_aux_loss = epoch_aux_loss / total_samples
        move_acc = correct_moves / total_samples

        print(f"\n--- Epoch {epoch} Train Summary ---")
        print(f"Loss: {avg_loss:.4f} | Policy Loss: {avg_pol_loss:.4f} | Value Loss: {avg_val_loss:.4f} | Aux Loss: {avg_aux_loss:.4f}")
        print(f"Move Prediction Accuracy: {move_acc * 100:.2f}%")

        return {"loss": avg_loss, "policy_loss": avg_pol_loss, "value_loss": avg_val_loss, "accuracy": move_acc}

    def val_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_policy_loss = 0.0
        val_value_loss = 0.0
        correct_moves = 0
        total_samples = 0

        device_type = "xpu" if "xpu" in str(self.device) else ("cuda" if "cuda" in str(self.device) else "cpu")
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device, non_blocking=True)
                target_policy = batch["policy"].to(self.device, non_blocking=True)
                target_value = batch["value"].to(self.device, non_blocking=True)
                threat_target = batch["threat_target"].to(self.device, non_blocking=True)
                edge_threat_target = batch["edge_threat_target"].to(self.device, non_blocking=True)
                dtw_target = batch["dtw_target"].to(self.device, non_blocking=True)
                legal_move_target = batch["legal_move_target"].to(self.device, non_blocking=True)
                target_policy = self.shape_policy_target(target_policy, legal_move_target)

                targets = {
                    "policy": target_policy,
                    "value": target_value,
                    "threat_target": threat_target,
                    "edge_threat_target": edge_threat_target,
                    "dtw_target": dtw_target,
                    "legal_move_target": legal_move_target
                }

                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                    outputs = self.model(features)
                    # Apply Temperature scaling to policy logits to control policy smoothing
                    if self.temperature != 1.0:
                        outputs["policy_logits"] = outputs["policy_logits"] / self.temperature
                    loss_dict = self.criterion(outputs, targets)
                
                batch_size = features.size(0)
                val_loss += loss_dict["loss"].item() * batch_size
                val_policy_loss += loss_dict["policy_loss"].item() * batch_size
                val_value_loss += loss_dict["value_loss"].item() * batch_size

                pred_moves = outputs["policy_logits"].argmax(dim=-1)
                gt_moves = batch["policy"].view(batch_size, -1).argmax(dim=-1).to(self.device)
                correct_moves += (pred_moves == gt_moves).sum().item()
                total_samples += batch_size

        avg_loss = val_loss / total_samples
        avg_pol_loss = val_policy_loss / total_samples
        avg_val_loss = val_value_loss / total_samples
        move_acc = correct_moves / total_samples

        print(f"\n--- Epoch {epoch} Validation Summary ---")
        print(f"Val Loss: {avg_loss:.4f} | Val Policy Loss: {avg_pol_loss:.4f} | Val Value Loss: {avg_val_loss:.4f}")
        print(f"Val Move Prediction Accuracy: {move_acc * 100:.2f}%")

        return {"loss": avg_loss, "policy_loss": avg_pol_loss, "value_loss": avg_val_loss, "accuracy": move_acc}

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        
        # Save latest
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, "latest_model.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved new best model checkpoint to {best_path}")

    def update_temperature(self, epoch):
        """
        Implements temperature scaling scheduling (Stage 3.2):
        Early: tau = 1.0, Mid: tau = 0.7, Late: tau = 0.5
        """
        total_epochs = self.config.NUM_EPOCHS
        if epoch < int(total_epochs * 0.3):
            self.temperature = 1.0
        elif epoch < int(total_epochs * 0.7):
            self.temperature = 0.7
        else:
            self.temperature = 0.5
        
        print(f"Softmax Temperature updated to {self.temperature}")

    def shape_policy_target(self, target_policy, legal_move_target):
        if not self.enable_top_k_policy:
            return target_policy

        batch_size = target_policy.size(0)
        flat_policy = target_policy.view(batch_size, -1)
        flat_legal = legal_move_target.view(batch_size, -1).bool()
        if self.policy_top_k <= 0 or self.policy_top_k >= flat_policy.size(1):
            return self.normalize_policy_target(flat_policy, flat_legal).view_as(target_policy)

        masked_policy = flat_policy.masked_fill(~flat_legal, 0.0)
        normalized = self.normalize_policy_target(masked_policy, flat_legal)
        legal_counts = flat_legal.sum(dim=1)
        top_k = min(self.policy_top_k, int(legal_counts.max().item()))
        if top_k <= 0:
            return normalized.view_as(target_policy)

        top_values, top_indices = torch.topk(normalized, k=top_k, dim=1)
        shaped = torch.zeros_like(normalized)
        shaped.scatter_(1, top_indices, top_values)

        if self.policy_top_k_floor > 0:
            floor_mask = flat_legal & (shaped <= 0)
            shaped = shaped + floor_mask.to(shaped.dtype) * self.policy_top_k_floor

        return self.normalize_policy_target(shaped, flat_legal).view_as(target_policy)

    @staticmethod
    def normalize_policy_target(flat_policy, flat_legal):
        masked = flat_policy.masked_fill(~flat_legal, 0.0)
        total = masked.sum(dim=1, keepdim=True)
        uniform = flat_legal.to(masked.dtype) / flat_legal.sum(dim=1, keepdim=True).clamp(min=1)
        return torch.where(total > 0, masked / total.clamp(min=1e-8), uniform)
