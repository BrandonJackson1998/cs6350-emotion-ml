# Quick Reference Cheatsheet 📋

Essential commands and information for the cs6350-emotion-ml project.

## 🚀 First Time Setup

```bash
# 1. Clone
git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
cd cs6350-emotion-ml

# 2. Install dependencies
make install          # Linux: auto install everything
make install-deb-mac  # macOS: install system deps
make install-pip      # Install Python packages

# 3. Download dataset from https://www.kaggle.com/datasets/msambare/fer2013
# Extract to data/train/ and data/test/

# 4. Verify setup
make verify

# 5. Run benchmark
make benchmark
```

## 💻 Daily Commands

```bash
# Activate environment
source .virtual_environment/bin/activate

# Run experiments
make benchmark                    # Standard benchmark
python run_custom_experiment.py   # Custom experiment
python my_experiment.py           # Your custom script

# Check results
ls experiments/
cat experiments/*/classification_report.txt

# Deactivate when done
deactivate
```

## 📁 File Structure

```
cs6350-emotion-ml/
├── src/benchmark.py              # Main code
├── data/train/                   # Training images
├── data/test/                    # Test images
├── experiments/                  # Results output
├── requirements.txt              # Python packages
├── verify_setup.py               # Setup checker
├── run_custom_experiment.py      # Example experiment
└── .virtual_environment/         # Virtual env (don't commit)
```

## 🔧 Make Commands

| Command | Purpose |
|---------|---------|
| `make help` | Show all commands |
| `make install` | Full installation |
| `make verify` | Check setup |
| `make benchmark` | Run benchmark |

## 🐍 Python Quick Tests

```bash
# Check Python version
python3 --version

# Test imports
python -c "import torch, transformers; print('OK')"

# Test with virtual env
source .virtual_environment/bin/activate
python verify_setup.py
```

## 📊 Experiment Configuration

```python
from src.benchmark import create_experiment_config, run_experiment

config = create_experiment_config(
    experiment_name="my_test",
    sampling_weights={
        'angry': 1.0,
        'disgust': 1.0,
        'fear': 2.0,      # 2x weight
        'happy': 1.0,
        'neutral': 1.0,
        'sad': 2.0,       # 2x weight
        'surprise': 1.0
    },
    num_epochs=5,          # Training iterations
    sample_per_class=100,  # Images per emotion
    batch_size=32,         # Batch size
    learning_rate=2e-5     # Learning rate
)

output_dir, accuracy = run_experiment(config)
```

## 🎯 Emotions Detected

1. Angry 😠
2. Disgust 🤢
3. Fear 😨
4. Happy 😊
5. Neutral 😐
6. Sad 😢
7. Surprise 😲

## 📈 Understanding Results

**Files in `experiments/[name]_[timestamp]/`:**
- `best_model.pt` - Best trained model
- `classification_report.txt` - Detailed metrics
- `confusion_matrix_benchmark.png` - Visual confusion
- `training_history.png` - Learning curves
- `experiment_config.json` - Configuration used

**Key Metrics:**
- **Accuracy**: % correct overall
- **Precision**: % correct of positive predictions
- **Recall**: % detected of actual positives
- **F1-Score**: Balanced metric

## 🐛 Quick Fixes

```bash
# Can't find module?
source .virtual_environment/bin/activate

# Missing packages?
pip install -r requirements.txt

# Dataset missing?
ls data/train/  # Should show 7 emotion folders

# Python version wrong?
python3 --version  # Need 3.12+

# Virtual env issues?
make install-pip  # Recreate it
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `QUICKSTART.md` | Fast 5-min setup |
| `FAQ.md` | Common questions |
| `WORKFLOW.md` | Detailed workflows |
| `CONTRIBUTING.md` | How to contribute |
| `PROJECT_ROADMAP.md` | Future plans |
| `CHEATSHEET.md` | This file! |

## 🔗 Important Links

- **Dataset**: https://www.kaggle.com/datasets/msambare/fer2013
- **PyTorch Docs**: https://pytorch.org/docs/
- **Transformers**: https://huggingface.co/docs/transformers/
- **Model Used**: https://huggingface.co/dima806/facial_emotions_image_detection

## ⚡ Performance Tips

```python
# Faster testing (lower accuracy)
config = create_experiment_config(
    experiment_name="quick_test",
    num_epochs=1,
    sample_per_class=20
)

# Better accuracy (slower)
config = create_experiment_config(
    experiment_name="deep_training",
    num_epochs=20,
    sample_per_class=200
)
```

## 🎓 Typical Workflow

```
1. Activate environment
2. Create/modify experiment script
3. Run experiment
4. Check results in experiments/
5. Adjust parameters
6. Repeat 3-5
7. Document findings
```

## 🆘 Help Resources

1. Check `FAQ.md` for common issues
2. Run `make verify` to check setup
3. Review error messages carefully
4. Search GitHub issues
5. Create new issue with details

## 💡 Pro Tips

- ✅ Always activate virtual environment first
- ✅ Start with small experiments (few epochs, few samples)
- ✅ Use `make verify` after any setup changes
- ✅ Save successful configurations
- ✅ Document experiment results
- ✅ Commit working code regularly

## 🚫 Common Mistakes

- ❌ Forgetting to activate virtual environment
- ❌ Not downloading dataset
- ❌ Running long experiments without testing first
- ❌ Committing virtual environment folder
- ❌ Modifying dataset files
- ❌ Hardcoding paths

---

**Quick Help**: Run `make help` or see `README.md` for detailed info!
