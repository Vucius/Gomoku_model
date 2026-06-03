import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.config import GomokuConfig
from src.model import GomokuNet
from src.dataset import GomokuDataset, TacticalEdgeBatchSampler
from src.trainer import GomokuTrainer

def main():
    parser = argparse.ArgumentParser(description="Gomoku Base Neural Network Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--teacher", type=str, default=None, help="Path to teacher model checkpoint (.pt)")
    parser.add_argument("--teacher_policy_weight", type=float, default=None, help="Teacher soft-policy blend weight")
    parser.add_argument("--teacher_value_weight", type=float, default=None, help="Teacher value blend weight")
    parser.add_argument("--policy_top_k", type=int, default=None, help="Keep only top-k legal policy target moves")
    parser.add_argument("--disable_top_k_policy", action="store_true", help="Disable top-k policy target shaping")
    parser.add_argument("--label_smoothing", type=float, default=None, help="Policy label smoothing epsilon")
    parser.add_argument("--entropy_alpha", type=float, default=None, help="Policy entropy regularization weight")
    parser.add_argument("--no_psq", action="store_true", help="Disable loading .psq dataset")
    parser.add_argument("--no_npy", action="store_true", help="Disable loading .npy dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint_prefix", type=str, default="base_model", help="Prefix for checkpoint filenames")
    parser.add_argument("--res_blocks", type=int, default=None, help="Override number of residual blocks")
    parser.add_argument("--hidden_channels", type=int, default=None, help="Override number of hidden channels")
    args = parser.parse_args()

    # 1. Initialize configuration
    config = GomokuConfig()
    
    # Override defaults with CLI args
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.teacher_policy_weight is not None:
        config.TEACHER_POLICY_WEIGHT = args.teacher_policy_weight
    if args.teacher_value_weight is not None:
        config.TEACHER_VALUE_WEIGHT = args.teacher_value_weight
    if args.policy_top_k is not None:
        config.POLICY_TOP_K = args.policy_top_k
    if args.disable_top_k_policy:
        config.ENABLE_TOP_K_POLICY = False
    if args.label_smoothing is not None:
        config.LABEL_SMOOTHING_EPS = args.label_smoothing
    if args.entropy_alpha is not None:
        config.POLICY_ENTROPY_ALPHA = args.entropy_alpha
    if args.res_blocks is not None:
        config.NUM_RES_BLOCKS = args.res_blocks
    if args.hidden_channels is not None:
        config.HIDDEN_CHANNELS = args.hidden_channels

    if not 0.0 <= config.TEACHER_POLICY_WEIGHT <= 1.0:
        raise ValueError("--teacher_policy_weight must be in [0, 1]")
    if not 0.0 <= config.TEACHER_VALUE_WEIGHT <= 1.0:
        raise ValueError("--teacher_value_weight must be in [0, 1]")
    if config.POLICY_TOP_K <= 0:
        raise ValueError("--policy_top_k must be positive")
    if config.POLICY_TOP_K_FLOOR < 0:
        raise ValueError("POLICY_TOP_K_FLOOR must be non-negative")
    if not 0.0 <= config.LABEL_SMOOTHING_EPS <= 1.0:
        raise ValueError("--label_smoothing must be in [0, 1]")
    if config.POLICY_ENTROPY_ALPHA < 0:
        raise ValueError("--entropy_alpha must be non-negative")

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    print("=== Gomoku Base Neural Network Training Setup ===")
    print(f"Board Size          : {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"Device              : {config.DEVICE}")
    print(f"Epochs              : {config.NUM_EPOCHS}")
    print(f"Batch Size          : {config.BATCH_SIZE}")
    print(f"Learning Rate       : {config.LEARNING_RATE}")
    print(f"Res Blocks          : {config.NUM_RES_BLOCKS}")
    print(f"Hidden Channels     : {config.HIDDEN_CHANNELS}")
    print(f"Checkpoint Prefix   : {args.checkpoint_prefix}")
    print(f"Policy Top-k        : {config.POLICY_TOP_K if config.ENABLE_TOP_K_POLICY else 'disabled'}")
    print(f"Label Smoothing     : {config.LABEL_SMOOTHING_EPS}")
    print(f"Entropy Alpha       : {config.POLICY_ENTROPY_ALPHA}")
    print(f"Teacher Weights     : policy={config.TEACHER_POLICY_WEIGHT}, value={config.TEACHER_VALUE_WEIGHT}")
    print(f"Use PSQ Dataset     : {not args.no_psq}")
    print(f"Use NPY Dataset     : {not args.no_npy}")
    print("============================================")

    # 2. Initialize Datasets & DataLoaders
    print("\nLoading datasets...")
    train_dataset = GomokuDataset(
        config=config,
        is_train=True,
        use_psq=not args.no_psq,
        use_npy=not args.no_npy
    )
    val_dataset = GomokuDataset(
        config=config,
        is_train=False,
        use_psq=not args.no_psq,
        use_npy=not args.no_npy
    )
    
    print(f"Train samples count : {len(train_dataset)}")
    print(f"Val samples count   : {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Error: Train dataset is empty! Verify dataset files are in place.")

    # Custom sampler for edge-rebalancing batch composition
    train_sampler = TacticalEdgeBatchSampler(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        drop_last=True
    )
    
    # Check if GPU device (XPU or CUDA) is available for pin_memory
    gpu_available = torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True if gpu_available else False,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if gpu_available else False,
        persistent_workers=True
    )

    # 3. Instantiate Student Model
    print("\nInitializing student model...")
    model = GomokuNet(config)
    
    # 4. (Optional) Load Teacher Model for Knowledge Distillation
    teacher_model = None
    if args.teacher is not None:
        if os.path.exists(args.teacher):
            print(f"Loading teacher model from {args.teacher}...")
            teacher_model = GomokuNet(config)
            checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
            teacher_model.load_state_dict(checkpoint["model_state_dict"])
            print("Teacher model loaded successfully!")
        else:
            print(f"Warning: Teacher checkpoint not found at {args.teacher}. Running standard training.")

    # 5. Initialize Trainer
    trainer = GomokuTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_model=teacher_model,
        checkpoint_prefix=args.checkpoint_prefix
    )

    # 6. Training Loop
    print("\nStarting training loop...")
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(f"Successfully resumed from epoch {checkpoint['epoch']} with validation loss {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint not found at {args.resume}. Starting training from scratch.")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        trainer.update_temperature(epoch)
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.val_epoch(epoch)
        trainer.scheduler.step()
        
        val_loss = val_metrics["loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        trainer.save_checkpoint(epoch, val_loss, is_best=is_best)

    print("\nTraining completed successfully!")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
