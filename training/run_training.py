"""
Entry point for AlphaZero chess training.

Usage:
    # Start fresh training
    python -m training.run_training

    # Resume from checkpoint
    python -m training.run_training --resume

    # Quick test with small config
    python -m training.run_training --quick

    # Export best model to ONNX after training
    python -m training.run_training --export-only
"""

import argparse
from pathlib import Path
from .train import Trainer, TrainingConfig
from .export_onnx import export_to_onnx


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained checkpoint to initialize from (e.g., checkpoints/best.pt from supervised training)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with reduced settings"
    )
    parser.add_argument(
        "--export-only", action="store_true", help="Only export best checkpoint to ONNX"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory for checkpoints"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of training iterations"
    )
    parser.add_argument(
        "--games", type=int, default=50, help="Self-play games per iteration"
    )
    parser.add_argument(
        "--simulations", type=int, default=200, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker processes for self-play",
    )
    args = parser.parse_args()

    if args.export_only:
        checkpoint_path = f"{args.checkpoint_dir}/best.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = f"{args.checkpoint_dir}/latest.pt"
        Path("game/static/engine").mkdir(parents=True, exist_ok=True)
        export_to_onnx(checkpoint_path, "game/static/engine/model.onnx")
        return

    if args.quick:
        config = TrainingConfig(
            games_per_iteration=5,
            num_simulations=50,
            batch_size=32,
            epochs_per_iteration=2,
            eval_games=4,
            eval_simulations=25,
            num_iterations=3,
            checkpoint_dir=args.checkpoint_dir,
        )
    elif args.pretrained:
        # Fine-tuning config: lower LR to preserve supervised knowledge
        config = TrainingConfig(
            games_per_iteration=args.games,
            num_simulations=args.simulations,
            learning_rate=0.0002,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=3,
            num_iterations=args.iterations,
            parallel_games=args.workers,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        config = TrainingConfig(
            games_per_iteration=args.games,
            num_simulations=args.simulations,
            num_iterations=args.iterations,
            parallel_games=args.workers,
            checkpoint_dir=args.checkpoint_dir,
        )

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint("latest")
    elif args.pretrained:
        trainer.load_pretrained(args.pretrained)

    trainer.run()

    # Export best model to ONNX
    print("\nExporting best model to ONNX...")
    checkpoint_path = f"{args.checkpoint_dir}/best.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = f"{args.checkpoint_dir}/latest.pt"
    if Path(checkpoint_path).exists():
        Path("game/static/engine").mkdir(parents=True, exist_ok=True)
        export_to_onnx(checkpoint_path, "game/static/engine/model.onnx")
        print("Done! Model ready for browser deployment.")
    else:
        print("No checkpoint found to export.")


if __name__ == "__main__":
    main()
