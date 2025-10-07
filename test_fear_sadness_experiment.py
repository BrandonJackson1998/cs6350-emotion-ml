#!/usr/bin/env python3
"""
Test experiment with 100 epochs focusing on fear and sadness emotions.
This will help us understand how higher sampling weights affect these specific emotions.
"""

from src.benchmark import create_experiment_config, run_experiment

def main():
    """Run experiment with higher weights for fear and sadness"""
    
    print("🧪 Testing Fear & Sadness Focus Experiment")
    print("=" * 60)
    print("Configuration:")
    print("- 100 epochs")
    print("- Higher sampling weights for fear and sadness")
    print("- Standard weights for other emotions")
    print("=" * 60)
    
    # Create experiment configuration
    config = create_experiment_config(
        experiment_name="fear_sadness_focus_100epochs",
        sampling_weights={
            'angry': 1.0,      # Standard weight
            'disgust': 1.0,    # Standard weight
            'fear': 3.0,       # 3x higher weight for fear
            'happy': 1.0,      # Standard weight
            'neutral': 1.0,    # Standard weight
            'sad': 3.0,        # 3x higher weight for sadness
            'surprise': 1.0    # Standard weight
        },
        num_epochs=100,        # 100 epochs as requested
        sample_per_class=100,  # Keep reasonable dataset size
        learning_rate=2e-5,    # Standard learning rate
        batch_size=32          # Standard batch size
    )
    
    print(f"\n📊 Experiment Details:")
    print(f"   Name: {config['experiment_name']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Samples per Class: {config['sample_per_class']}")
    print(f"\n🎯 Sampling Weights:")
    for emotion, weight in config['sampling_weights'].items():
        print(f"   {emotion:10s}: {weight:4.1f}x")
    
    print(f"\n🚀 Starting experiment...")
    print("=" * 60)
    
    try:
        output_dir, best_acc = run_experiment(config)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"   📈 Best validation accuracy: {best_acc:.4f}")
        print(f"   📁 Output directory: {output_dir}")
        print(f"\n📋 Files generated:")
        print(f"   - {output_dir}/best_model.pt (best performing model)")
        print(f"   - {output_dir}/epoch_*_model.pt (all epoch models)")
        print(f"   - {output_dir}/confusion_matrix_benchmark.png")
        print(f"   - {output_dir}/training_history.png")
        print(f"   - {output_dir}/classification_report.txt")
        print(f"   - {output_dir}/experiment_config.json")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
