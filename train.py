import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.config import GomokuConfig
from src.model import GomokuNet
from src.dataset import GomokuDataset, TacticalEdgeBatchSampler
from src.trainer import GomokuTrainer

def main():
    parser = argparse.ArgumentParser(description="Gomoku Neural Network Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--teacher", type=str, default=None, help="Path to teacher model checkpoint (.pt)")
    parser.add_argument("--no_psq", action="store_true", help="Disable loading .psq dataset")
    parser.add_argument("--no_npy", action="store_true", help="Disable loading .npy dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    print("=== Gomoku Neural Network Training Setup ===")
    print(f"Board Size          : {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"Device              : {config.DEVICE}")
    print(f"Epochs              : {config.NUM_EPOCHS}")
    print(f"Batch Size          : {config.BATCH_SIZE}")
    print(f"Learning Rate       : {config.LEARNING_RATE}")
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4 if os.name != 'nt' else 0,  # Multi-processing has quirks on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True if torch.cuda.is_available() else False
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
            checkpoint = torch.load(args.teacher, map_location="cpu")
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
        teacher_model=teacher_model
    )

    # 6. Training Loop
    print("\nStarting training loop...")
    best_val_loss = float("inf")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Update temperature scaling
        trainer.update_temperature(epoch)
        
        # Train one epoch
        train_metrics = trainer.train_epoch(epoch)
        
        # Validate one epoch
        val_metrics = trainer.val_epoch(epoch)
        
        # Adjust learning rate
        trainer.scheduler.step()
        
        # Checkpoint checks
        val_loss = val_metrics["loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        trainer.save_checkpoint(epoch, val_loss, is_best=is_best)

    print("\nTraining completed successfully!")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
