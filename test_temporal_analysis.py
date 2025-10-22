#!/usr/bin/env python3
"""
Test Script for Temporal Emotion Analysis System
Validates core functionality and integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
import json
from PIL import Image
import numpy as np

def create_test_frames(output_dir, num_frames=5):
    """Create simple test frames for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_frames} test frames in {output_dir}")
    
    # Create simple colored frames to test emotion analysis
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255)   # Magenta
    ]
    
    for i in range(num_frames):
        # Create a simple colored square image
        color = colors[i % len(colors)]
        img = Image.new('RGB', (224, 224), color=color)
        
        # Add some simple pattern
        pixels = img.load()
        for x in range(50, 174):
            for y in range(50, 174):
                # Create a simple face-like pattern
                if ((x-112)**2 + (y-100)**2) < 20**2:  # Head circle
                    pixels[x, y] = (200, 200, 200)
                elif ((x-100)**2 + (y-90)**2) < 5**2:  # Left eye
                    pixels[x, y] = (0, 0, 0)
                elif ((x-124)**2 + (y-90)**2) < 5**2:  # Right eye
                    pixels[x, y] = (0, 0, 0)
                elif ((x-112)**2 + (y-120)**2) < 10**2:  # Mouth
                    pixels[x, y] = (100, 100, 100)
        
        img.save(os.path.join(output_dir, f"{i}.jpg"))
    
    print(f"✓ Created {num_frames} test frames")

def test_temporal_analyzer():
    """Test the temporal emotion analyzer"""
    print("🧪 Testing Temporal Emotion Analysis System")
    print("=" * 50)
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_frames_dir = os.path.join(temp_dir, "test_frames")
        output_dir = os.path.join(temp_dir, "output")
        
        # Create test frames
        create_test_frames(test_frames_dir, 5)
        
        # Import the analyzer
        try:
            from src.temporal_emotion_analyzer import TemporalEmotionAnalyzer, create_default_config
            print("✓ Successfully imported TemporalEmotionAnalyzer")
        except ImportError as e:
            print(f"❌ Failed to import analyzer: {e}")
            return False
        
        try:
            # Initialize analyzer
            config = create_default_config()
            config['batch_size'] = 2  # Small batch for testing
            
            analyzer = TemporalEmotionAnalyzer(config=config)
            print("✓ Analyzer initialized")
            
            # Load model
            analyzer.load_model()
            print("✓ Model loaded")
            
            # Load frames
            frames = analyzer.load_sequential_frames(test_frames_dir)
            print(f"✓ Loaded {len(frames)} frames")
            
            # Predict emotions
            emotions = analyzer.predict_frame_emotions(frames)
            print(f"✓ Predicted emotions: {emotions.shape}")
            
            # Calculate deltas
            deltas = analyzer.calculate_emotion_deltas(emotions)
            print(f"✓ Calculated deltas: {deltas.shape}")
            
            # Generate plots
            analyzer.generate_temporal_plots(output_dir)
            print("✓ Generated plots")
            
            # Save data
            analyzer.save_data(output_dir)
            print("✓ Saved data files")
            
            # Verify output files
            expected_files = [
                'emotion_probabilities.csv',
                'emotion_deltas.csv',
                'frame_information.csv',
                'analysis_summary.json',
                'analysis_config.json',
                'emotion_probabilities_temporal.png',
                'emotion_deltas_temporal.png',
                'emotion_statistics.png'
            ]
            
            missing_files = []
            for file in expected_files:
                if not os.path.exists(os.path.join(output_dir, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"❌ Missing files: {missing_files}")
                return False
            else:
                print("✓ All expected files generated")
            
            # Test pattern analysis
            patterns = analyzer.analyze_temporal_patterns()
            print(f"✓ Pattern analysis complete: {len(patterns)} emotions analyzed")
            
            print("\n🎉 All tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_command_line_interface():
    """Test command line interface"""
    print("\n🧪 Testing Command Line Interface")
    print("=" * 50)
    
    # Test help message
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, 
            "src/temporal_emotion_analyzer.py", 
            "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Help message displayed correctly")
            return True
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Temporal Emotion Analysis Tests")
    print("=" * 60)
    
    tests = [
        ("Core Functionality", test_temporal_analyzer),
        ("Command Line Interface", test_command_line_interface)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())