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
        self.teacher_alpha = config.TEACHER_POLICY_WEIGHT

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

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Train]")
        for batch in loop:
            # Move data to device
            features = batch["features"].to(self.device)
            target_policy = batch["policy"].to(self.device)
            target_value = batch["value"].to(self.device)
            threat_target = batch["threat_target"].to(self.device)
            edge_threat_target = batch["edge_threat_target"].to(self.device)
            dtw_target = batch["dtw_target"].to(self.device)
            legal_move_target = batch["legal_move_target"].to(self.device)

            # Knowledge Distillation (Stage 3)
            # If teacher model is available, mix standard targets with teacher predictions
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(features)
                    teacher_policy_soft = torch.softmax(teacher_outputs["policy_logits"], dim=-1)
                    teacher_value_soft = teacher_outputs["value"]
                
                # Reshape soft policy to match targets
                target_policy_flat = target_policy.view(target_policy.size(0), -1)
                mixed_policy = (1.0 - self.teacher_alpha) * target_policy_flat + self.teacher_alpha * teacher_policy_soft
                target_policy = mixed_policy.view_as(target_policy)

                mixed_value = (1.0 - self.teacher_alpha) * target_value + self.teacher_alpha * teacher_value_soft
                target_value = mixed_value

            targets = {
                "policy": target_policy,
                "value": target_value,
                "threat_target": threat_target,
                "edge_threat_target": edge_threat_target,
                "dtw_target": dtw_target,
                "legal_move_target": legal_move_target
            }

            self.optimizer.zero_grad()
            
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

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                target_policy = batch["policy"].to(self.device)
                target_value = batch["value"].to(self.device)
                threat_target = batch["threat_target"].to(self.device)
                edge_threat_target = batch["edge_threat_target"].to(self.device)
                dtw_target = batch["dtw_target"].to(self.device)
                legal_move_target = batch["legal_move_target"].to(self.device)

                targets = {
                    "policy": target_policy,
                    "value": target_value,
                    "threat_target": threat_target,
                    "edge_threat_target": edge_threat_target,
                    "dtw_target": dtw_target,
                    "legal_move_target": legal_move_target
                }

                outputs = self.model(features)
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
