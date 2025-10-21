# Quick Start Guide 🚀

This is a condensed guide for getting started quickly. For detailed explanations, see the [main README](README.md).

## Prerequisites

- Python 3.12
- Git
- 5GB free disk space

## Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
cd cs6350-emotion-ml

# 2. Install system dependencies
# Linux:
make install-deb
# macOS:
make install-deb-mac

# 3. Install Python packages
make install-pip
```

## Download Dataset (Manual - 10 minutes)

1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download the dataset (requires Kaggle account)
3. Extract to project:
   ```bash
   unzip ~/Downloads/archive.zip -d /tmp/fer2013
   cp -r /tmp/fer2013/train ./data/
   cp -r /tmp/fer2013/test ./data/
   ```

## Run Benchmark (15-30 minutes)

```bash
# Activate environment
source .virtual_environment/bin/activate

# Run benchmark
make benchmark
```

## Verify Setup (Recommended)

Before running experiments, verify your setup:

```bash
source .virtual_environment/bin/activate
python verify_setup.py
```

Should show all checks passing ✅.

## Check Results

Results are saved to `./experiments/` directory with:
- Trained models (`.pt` files)
- Confusion matrix (`.png`)
- Classification report (`.txt`)
- Training history (`.png` and `.csv`)

## Common Commands

```bash
# Activate virtual environment
source .virtual_environment/bin/activate

# Run benchmark
make benchmark

# Run custom experiment
python run_custom_experiment.py

# Run fear/sadness focused experiment
python test_fear_sadness_experiment.py

# Deactivate virtual environment
deactivate
```

## Troubleshooting

**Virtual environment not found?**
```bash
make install-pip
```

**Import errors?**
```bash
source .virtual_environment/bin/activate
pip install --upgrade -r requirements.txt
```

**Dataset not found?**
- Check `data/train/` and `data/test/` directories exist
- Verify 7 emotion subdirectories in each
- See dataset download section in main README

## Next Steps

- Review results in `experiments/` directory
- Modify experiments in `run_custom_experiment.py`
- Read the full [README](README.md) for detailed explanations
- Check [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for future goals

## Need Help?

See the [Troubleshooting section](README.md#-troubleshooting) in the main README or open an issue on GitHub.
