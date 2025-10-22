#!/usr/bin/env python3
"""
Generic Emotion Recognition Experiment Runner

This script provides a flexible interface for running emotion recognition experiments
with custom training distributions and checkpoint resumption capabilities.

Usage:
    # Run a new experiment
    python3 scripts/run_experiment.py --name "my_experiment" --epochs 20 --sad 5.0 --happy 0.5

    # Resume from checkpoint with new distribution
    python3 scripts/run_experiment.py --resume ./experiments/old_experiment/checkpoint_epoch_5.pt --epochs 10 --happy 4.0 --sad 0.3

    # Quick start with balanced training
    python3 scripts/run_experiment.py --name "balanced" --epochs 15
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from benchmark import run_experiment, create_experiment_config

def list_available_checkpoints(experiments_dir="./experiments"):
    """List all available checkpoints in experiments directory"""
    checkpoints = []
    
    if not os.path.exists(experiments_dir):
        print(f"No experiments directory found at {experiments_dir}")
        return checkpoints
    
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
            
        # Look for checkpoint files
        for file in os.listdir(exp_path):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                checkpoint_path = os.path.join(exp_path, file)
                checkpoints.append({
                    'path': checkpoint_path,
                    'experiment': exp_dir,
                    'epoch': file.replace('checkpoint_epoch_', '').replace('.pt', ''),
                    'full_path': checkpoint_path
                })
    
    return sorted(checkpoints, key=lambda x: (x['experiment'], int(x['epoch'])))

def parse_weights(args):
    """Parse sampling weights from command line arguments."""
    weights = {
        'angry': getattr(args, 'angry', 1.0),
        'disgust': getattr(args, 'disgust', 1.0),
        'fear': getattr(args, 'fear', 1.0),
        'happy': getattr(args, 'happy', 1.0),
        'neutral': getattr(args, 'neutral', 1.0),
        'sad': getattr(args, 'sad', 1.0),
        'surprise': getattr(args, 'surprise', 1.0)
    }
    return weights

def main():
    parser = argparse.ArgumentParser(
        description="Generic Emotion Recognition Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New experiment with sadness focus
  python3 scripts/run_experiment.py --name "sadness_focus" --epochs 20 --sad 5.0 --happy 0.5

  # Resume with happiness focus
  python3 scripts/run_experiment.py --resume ./experiments/old/checkpoint_epoch_10.pt --epochs 10 --happy 4.0 --sad 0.3

  # Balanced training
  python3 scripts/run_experiment.py --name "balanced" --epochs 15

  # Fear and sadness focus
  python3 scripts/run_experiment.py --name "fear_sad" --epochs 25 --fear 3.0 --sad 3.0 --happy 0.3
        """
    )

    # Experiment configuration
    parser.add_argument('--name', type=str, required=False,
                       help='Experiment name (will be timestamped automatically)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume from')
    parser.add_argument('--list', action='store_true',
                       help='List available checkpoints and exit')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='dima806/facial_emotions_image_detection',
                       help='Model name to use (default: dima806/facial_emotions_image_detection)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='Number of samples per class (default: 100)')

    # Emotion sampling weights (all default to 1.0 for balanced training)
    parser.add_argument('--angry', type=float, default=1.0,
                       help='Sampling weight for angry emotion (default: 1.0)')
    parser.add_argument('--disgust', type=float, default=1.0,
                       help='Sampling weight for disgust emotion (default: 1.0)')
    parser.add_argument('--fear', type=float, default=1.0,
                       help='Sampling weight for fear emotion (default: 1.0)')
    parser.add_argument('--happy', type=float, default=1.0,
                       help='Sampling weight for happy emotion (default: 1.0)')
    parser.add_argument('--neutral', type=float, default=1.0,
                       help='Sampling weight for neutral emotion (default: 1.0)')
    parser.add_argument('--sad', type=float, default=1.0,
                       help='Sampling weight for sad emotion (default: 1.0)')
    parser.add_argument('--surprise', type=float, default=1.0,
                       help='Sampling weight for surprise emotion (default: 1.0)')

    # Additional options
    parser.add_argument('--description', type=str, default=None,
                       help='Custom description for the experiment')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda) (default: auto)')

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        print("📋 Available Checkpoints:")
        print("=" * 60)
        checkpoints = list_available_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
            return
        
        for i, cp in enumerate(checkpoints, 1):
            print(f"{i:2d}. {cp['experiment']} - Epoch {cp['epoch']}")
            print(f"    Path: {cp['path']}")
        return

    # Validate required arguments
    if not args.name and not args.resume:
        print("❌ Error: Either --name (for new experiment) or --resume (for resuming) is required")
        print("💡 Use --list to see available checkpoints")
        sys.exit(1)

    # Parse sampling weights
    sampling_weights = parse_weights(args)

    # Create experiment description
    if args.description:
        description = args.description
    elif args.resume:
        description = f"Resumed experiment with custom emotion focus"
    else:
        # Auto-generate description based on weights
        focus_emotions = [emotion for emotion, weight in sampling_weights.items() if weight > 1.5]
        if focus_emotions:
            description = f"Experiment with focus on: {', '.join(focus_emotions)}"
        else:
            description = "Balanced emotion recognition experiment"

    # Create experiment configuration
    config = create_experiment_config(
        experiment_name=args.name,
        sampling_weights=sampling_weights,
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        sample_per_class=args.samples_per_class,
        resume_from_checkpoint=args.resume,
        experiment_description=description
    )

    print(f"🎭 EMOTION RECOGNITION EXPERIMENT")
    print("=" * 50)
    print(f"📝 Experiment: {args.name}")
    print(f"📊 Epochs: {args.epochs}")
    if args.resume:
        print(f"🔄 Resuming from: {args.resume}")
    else:
        print("🚀 Starting new experiment")
    print(f"🎯 Model: {args.model}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Learning rate: {args.learning_rate}")
    print(f"🎲 Samples per class: {args.samples_per_class}")
    print()
    
    print("🎯 Sampling Weights:")
    for emotion, weight in sampling_weights.items():
        print(f"   {emotion:8}: {weight:4.1f}x")
    print()

    # Run the experiment
    try:
        output_dir, best_val_acc = run_experiment(config)
        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📈 Best validation accuracy: {best_val_acc:.4f}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
