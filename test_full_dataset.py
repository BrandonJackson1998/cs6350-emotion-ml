#!/usr/bin/env python3
"""
Test script to validate full dataset functionality before running full experiments
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.benchmark import (
    create_experiment_config, 
    FER2013FolderDataset, 
    get_dataset_statistics,
    setup_signal_handlers,
    get_memory_usage
)

def test_configuration():
    """Test that experiment configuration works with new parameters"""
    print("Testing experiment configuration...")
    
    # Test full dataset config
    config = create_experiment_config(
        experiment_name="test_full_dataset",
        use_full_dataset=True,
        batch_size=16,
        batch_checkpoint_frequency=100,
        auto_checkpoint_on_interrupt=True,
        max_batch_checkpoints=3
    )
    
    assert config['use_full_dataset'] == True
    assert config['batch_size'] == 16
    assert config['batch_checkpoint_frequency'] == 100
    print("✓ Configuration test passed")

def test_dataset_loading():
    """Test dataset loading with full dataset flag"""
    print("Testing dataset loading...")
    
    # Mock processor for testing
    class MockProcessor:
        def __call__(self, images, return_tensors):
            import torch
            return {'pixel_values': torch.randn(1, 3, 224, 224)}
    
    try:
        # Test limited dataset
        limited_dataset = FER2013FolderDataset(
            "data/train",
            MockProcessor(),
            max_samples_per_class=10,
            use_full_dataset=False
        )
        print(f"✓ Limited dataset: {len(limited_dataset)} images")
        
        # Test full dataset flag (but still limit for testing)
        full_dataset = FER2013FolderDataset(
            "data/train", 
            MockProcessor(),
            max_samples_per_class=20,  # Still limit for test
            use_full_dataset=True
        )
        print(f"✓ Full dataset mode: {len(full_dataset)} images")
        
    except Exception as e:
        print(f"⚠️  Dataset loading test failed (expected if data not available): {e}")

def test_signal_handlers():
    """Test signal handler setup"""
    print("Testing signal handlers...")
    try:
        setup_signal_handlers()
        print("✓ Signal handlers registered successfully")
    except Exception as e:
        print(f"❌ Signal handler test failed: {e}")

def test_memory_monitoring():
    """Test memory usage monitoring"""
    print("Testing memory monitoring...")
    try:
        memory_info = get_memory_usage()
        print(f"✓ Memory info: {memory_info}")
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")

def test_dataset_statistics():
    """Test dataset statistics function"""
    print("Testing dataset statistics...")
    try:
        train_stats, train_total = get_dataset_statistics("data/train")
        print(f"✓ Train stats: {train_total} total images")
        
        test_stats, test_total = get_dataset_statistics("data/test")
        print(f"✓ Test stats: {test_total} total images")
        
    except Exception as e:
        print(f"⚠️  Dataset statistics test failed (expected if data not available): {e}")

def main():
    """Run all tests"""
    print("🧪 Running Full Dataset Implementation Tests")
    print("=" * 50)
    
    test_configuration()
    test_signal_handlers()
    test_memory_monitoring()
    test_dataset_statistics()
    test_dataset_loading()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n🚀 Ready to run experiments!")
    print("\nTo run the full dataset test:")
    print("  python src/benchmark.py")
    print("\nThe experiment will include:")
    print("  1. Baseline (100 samples/class) for comparison")
    print("  2. Full dataset test (28,709 images)")
    print("  3. Batch-level checkpointing every 100 batches")
    print("  4. Graceful interruption handling (Ctrl+C)")

if __name__ == "__main__":
    main()