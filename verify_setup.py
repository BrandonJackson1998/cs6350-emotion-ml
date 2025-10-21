#!/usr/bin/env python3
"""
Setup Verification Script
Run this script to verify your development environment is correctly configured.
"""

import sys
import os
from pathlib import Path

def print_status(message, status):
    """Print status message with emoji"""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {message}")
    return status

def check_python_version():
    """Check if Python version is correct"""
    version = sys.version_info
    is_correct = version.major == 3 and version.minor >= 12
    print_status(
        f"Python version: {version.major}.{version.minor}.{version.micro} (Required: 3.12+)",
        is_correct
    )
    return is_correct

def check_imports():
    """Check if all required packages can be imported"""
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "PIL",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "sklearn",
        "tqdm"
    ]
    
    all_imported = True
    for package in packages:
        try:
            __import__(package)
            print_status(f"Package '{package}' can be imported", True)
        except ImportError as e:
            print_status(f"Package '{package}' import failed: {e}", False)
            all_imported = False
    
    return all_imported

def check_dataset():
    """Check if dataset is properly set up"""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Check train directory
    train_dir = Path("data/train")
    if not train_dir.exists():
        print_status("Training dataset directory exists", False)
        return False
    
    print_status("Training dataset directory exists", True)
    
    # Check test directory
    test_dir = Path("data/test")
    if not test_dir.exists():
        print_status("Test dataset directory exists", False)
        return False
    
    print_status("Test dataset directory exists", True)
    
    # Check all emotion subdirectories
    all_present = True
    for emotion in emotions:
        train_emotion_dir = train_dir / emotion
        test_emotion_dir = test_dir / emotion
        
        if train_emotion_dir.exists() and test_emotion_dir.exists():
            train_count = len(list(train_emotion_dir.glob("*.png"))) + len(list(train_emotion_dir.glob("*.jpg")))
            test_count = len(list(test_emotion_dir.glob("*.png"))) + len(list(test_emotion_dir.glob("*.jpg")))
            print_status(
                f"Emotion '{emotion}': train={train_count}, test={test_count}",
                True
            )
        else:
            print_status(f"Emotion '{emotion}' directory missing", False)
            all_present = False
    
    return all_present

def check_project_files():
    """Check if essential project files exist"""
    files = [
        "src/benchmark.py",
        "requirements.txt",
        "makefile",
        "README.md",
        "run_custom_experiment.py"
    ]
    
    all_exist = True
    for file in files:
        exists = Path(file).exists()
        print_status(f"File '{file}' exists", exists)
        if not exists:
            all_exist = False
    
    return all_exist

def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = sys.prefix != sys.base_prefix
    print_status("Running in virtual environment", in_venv)
    if not in_venv:
        print("   ⚠️  It's recommended to run this in a virtual environment")
        print("   Run: source .virtual_environment/bin/activate")
    return True  # Not a blocker

def main():
    """Main verification function"""
    print("=" * 60)
    print("🔍 CS6350 Emotion ML - Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Required Packages", check_imports),
        ("Project Files", check_project_files),
        ("Dataset", check_dataset),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n📋 Checking: {check_name}")
        print("-" * 60)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(f"Check failed with error: {e}", False)
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        print(f"{emoji} {check_name:30s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\n📝 Next steps:")
        print("   1. Run: make benchmark")
        print("   2. Check results in: ./experiments/")
        print("   3. Read QUICKSTART.md for more commands")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\n📚 Troubleshooting:")
        print("   1. Make sure you activated the virtual environment:")
        print("      source .virtual_environment/bin/activate")
        print("   2. Install dependencies:")
        print("      make install-pip")
        print("   3. Download the dataset (see README.md)")
        print("   4. Check the Troubleshooting section in README.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
