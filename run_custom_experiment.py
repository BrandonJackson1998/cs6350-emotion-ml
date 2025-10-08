#!/usr/bin/env python3
"""
Example script showing how to run custom experiments with different sampling weights.
This demonstrates the new experiment system capabilities.
"""

from src.benchmark import create_experiment_config, run_experiment

def main():
    """Run a custom experiment with specific sampling weights"""
    
    # Example 1: Focus on underrepresented emotions
    print("Running experiment focused on underrepresented emotions...")
    
    config = create_experiment_config(
        experiment_name="custom_underrepresented_focus",
        sampling_weights={
            'angry': 1.0,
            'disgust': 4.0,  # Much higher weight for very underrepresented class
            'fear': 2.0,     # Higher weight for underrepresented class
            'happy': 0.8,    # Slightly lower weight for overrepresented class
            'neutral': 1.0,
            'sad': 1.0,
            'surprise': 1.5  # Slightly higher weight
        },
        num_epochs=3,  # Shorter for testing
        sample_per_class=50  # Smaller dataset for faster testing
    )
    
    try:
        output_dir, best_acc = run_experiment(config)
        print(f"✅ Custom experiment completed!")
        print(f"   Best accuracy: {best_acc:.4f}")
        print(f"   Output directory: {output_dir}")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")

if __name__ == "__main__":
    main()
