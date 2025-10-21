# cs6350-emotion-ml 🎭

Detecting change in emotion given a time series of images using machine learning.

This project uses the FER-2013 dataset and pre-trained emotion detection models to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## 🚀 Quick Start

This guide will help you set up and run the project from scratch. No prior experience required!

> **📖 Looking for a faster guide?** Check out [QUICKSTART.md](QUICKSTART.md) for a condensed version with just the essential commands.

### Prerequisites

Before you begin, make sure you have:
- **Python 3.12** installed on your system
- **Git** installed
- **Internet connection** for downloading dependencies and datasets
- At least **5GB of free disk space** for the dataset

### Operating System Support

- ✅ Linux (Ubuntu/Debian) - Recommended
- ✅ macOS
- ⚠️ Windows (may require WSL or manual setup)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
cd cs6350-emotion-ml
```

### Step 2: Install System Dependencies

**For Linux (Ubuntu/Debian):**
```bash
make install-deb
```

This will install:
- `python3.12-venv` - Python virtual environment support
- `ffmpeg` - Required for processing audio files

**For macOS:**
```bash
make install-deb-mac
```

This will install:
- `python@3.12` - Python 3.12
- `ffmpeg` - Required for processing audio files

**For manual installation:**
- On Linux: `sudo apt install python3.12-venv ffmpeg`
- On macOS: `brew install python@3.12 ffmpeg`

### Step 3: Create Virtual Environment and Install Python Dependencies

**Automatic installation (recommended):**
```bash
make install-pip
```

**Manual installation:**
```bash
# Create virtual environment
python3.12 -m venv .virtual_environment

# Activate virtual environment
source .virtual_environment/bin/activate

# Install Python packages
pip install --upgrade -r requirements.txt
```

The installation will download and install:
- PyTorch (deep learning framework)
- Transformers (pre-trained models)
- scikit-learn (machine learning utilities)
- matplotlib, seaborn (visualization)
- And other required packages

**⏱️ Note:** Installation may take 5-10 minutes depending on your internet speed.

### Step 4: Download the Dataset

The project uses the FER-2013 dataset from Kaggle. You need to download it manually:

1. **Create a Kaggle account** (if you don't have one):
   - Go to https://www.kaggle.com/
   - Sign up for a free account

2. **Download the dataset**:
   - Visit: https://www.kaggle.com/datasets/msambare/fer2013
   - Click the "Download" button (you may need to accept the terms)
   - You'll get a file named `archive.zip` or similar

3. **Extract the dataset**:
   ```bash
   # Assuming you downloaded to ~/Downloads/archive.zip
   unzip ~/Downloads/archive.zip -d /tmp/fer2013
   
   # Copy to the project directory
   cp -r /tmp/fer2013/train ./data/
   cp -r /tmp/fer2013/test ./data/
   ```

4. **Verify the dataset structure**:
   ```bash
   ls data/train/
   # Should show: angry  disgust  fear  happy  neutral  sad  surprise
   
   ls data/test/
   # Should show: angry  disgust  fear  happy  neutral  sad  surprise
   ```

**Alternative: Using Kaggle API** (for advanced users):
```bash
# Install and configure Kaggle API
pip install kaggle
# Follow instructions at: https://github.com/Kaggle/kaggle-api#api-credentials

# Download dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d ./data/
```

## 🏃 Running the Project

### Activate Virtual Environment

Before running any commands, always activate the virtual environment:

```bash
source .virtual_environment/bin/activate
```

You should see `(.virtual_environment)` at the beginning of your terminal prompt.

### Run the Benchmark

To run the baseline benchmark with default settings (100 samples per class, 5 epochs):

```bash
make benchmark
```

**Or manually:**
```bash
source .virtual_environment/bin/activate
python -m src.benchmark
```

**⏱️ Expected runtime:** 15-30 minutes depending on your hardware (faster with GPU).

**What this does:**
- Loads 100 images per emotion class for training and testing
- Trains multiple emotion detection models with different configurations
- Generates evaluation metrics and visualizations
- Saves results to `./experiments/` directory

### Run Custom Experiments

To run a custom experiment with specific parameters:

```bash
source .virtual_environment/bin/activate
python run_custom_experiment.py
```

Or run a specific test experiment:

```bash
source .virtual_environment/bin/activate
python test_fear_sadness_experiment.py
```

**What this does:**
- Runs experiments with custom sampling weights
- Focuses on specific emotions (e.g., fear and sadness)
- Trains for a custom number of epochs
- Saves results to `./experiments/` directory

## 📊 Understanding the Output

After running the benchmark, you'll find results in the `experiments/` directory:

```
experiments/
└── baseline_20231007_123456/
    ├── best_model.pt                    # Best performing model
    ├── classification_report.txt        # Detailed metrics per emotion
    ├── confusion_matrix_benchmark.png   # Visual confusion matrix
    ├── training_history.png             # Loss and accuracy plots
    ├── training_history.csv             # Training data in CSV format
    └── experiment_config.json           # Experiment configuration
```

### Key Metrics Explained

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Of all predictions for an emotion, how many were correct
- **Recall**: Of all actual instances of an emotion, how many were detected
- **F1-Score**: Harmonic mean of precision and recall

## 🎯 First Run - Benchmark Results

### Model Performance
- **Overall Accuracy**: 61.14%
- **Training Setup**: 100 samples per class (700 total)
- **Validation Set**: 100 samples per class (700 total)
- **Epochs**: 5
- **Model**: dima806/facial_emotions_image_detection

### Classification Report

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.4390    | 0.5400 | 0.4843   | 100     |
| Disgust  | 0.8972    | 0.9600 | 0.9275   | 100     |
| Fear     | 0.3934    | 0.2400 | 0.2981   | 100     |
| Happy    | 0.8370    | 0.7700 | 0.8021   | 100     |
| Neutral  | 0.5088    | 0.5800 | 0.5421   | 100     |
| Sad      | 0.3981    | 0.4100 | 0.4039   | 100     |
| Surprise | 0.7800    | 0.7800 | 0.7800   | 100     |
| **Macro Avg** | **0.6076** | **0.6114** | **0.6054** | **700** |

### Key Insights

**Strong Performance:**
- 🎭 **Disgust**: 96% recall, 90% precision (best performing)
- 😊 **Happy**: 77% recall, 84% precision
- 😲 **Surprise**: 78% recall and precision

**Weak Performance:**
- 😨 **Fear**: 24% recall, 39% precision (worst performing)
- 😢 **Sad**: 41% recall, 40% precision
- 😠 **Angry**: 54% recall, 44% precision

**Observations:**
- Model excels at distinct expressions (disgust, surprise, happy)
- Struggles with subtle or ambiguous emotions (fear, sad, angry)
- Overall performance (61.14%) significantly exceeds random baseline (14.3%)

## 🛠️ Available Make Commands

The project includes a Makefile with convenient commands:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Complete installation (system packages + Python packages) |
| `make install-deb` | Install system packages (Linux/Ubuntu) |
| `make install-deb-mac` | Install system packages (macOS) |
| `make install-pip` | Create virtual environment and install Python packages |
| `make verify` | Verify setup is correct and all dependencies are installed |
| `make benchmark` | Run the benchmark experiment |

**Example usage:**
```bash
# Show help
make help

# Install everything (Linux)
make install

# Just run the benchmark
make benchmark
```

## 📁 Project Structure

```
cs6350-emotion-ml/
├── data/                          # Dataset directory
│   ├── train/                     # Training images (organized by emotion)
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                      # Test images (organized by emotion)
│       └── (same structure as train)
│
├── src/                           # Source code
│   └── benchmark.py              # Main benchmark and training code
│
├── experiments/                   # Output directory for experiment results
│   └── (generated after running experiments)
│
├── requirements.txt              # Python dependencies
├── makefile                      # Build automation commands
├── run_custom_experiment.py      # Example custom experiment script
├── test_fear_sadness_experiment.py  # Test experiment focusing on specific emotions
└── README.md                     # This file
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "No module named 'torch'"

**Solution:** Make sure you activated the virtual environment:
```bash
source .virtual_environment/bin/activate
```

#### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'data/train'"

**Solution:** You need to download the dataset first. See Step 4 in the Installation section.

#### Issue: "CUDA out of memory" or slow training

**Solutions:**
- Reduce batch size in the configuration (default is 32)
- Reduce samples per class
- Close other applications using GPU memory
- If no GPU available, training will use CPU (slower but works)

#### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:** Make sure you're running commands from the project root directory:
```bash
cd /path/to/cs6350-emotion-ml
python -m src.benchmark
```

#### Issue: Python version mismatch

**Solution:** This project requires Python 3.12. Check your version:
```bash
python3 --version
```

If you have a different version, you may need to:
- Install Python 3.12
- Use `python3.12` explicitly in commands
- Recreate the virtual environment with Python 3.12

#### Issue: Permission denied when installing system packages

**Solution:** On Linux, use `sudo` for system package installation:
```bash
sudo apt install python3.12-venv ffmpeg
```

### Getting Help

If you encounter issues not covered here:
1. Check that all installation steps were completed
2. Ensure the virtual environment is activated
3. Verify the dataset is properly downloaded and extracted
4. Check the `experiments/` directory for error logs
5. Create an issue on the GitHub repository with:
   - Your operating system
   - Python version (`python3 --version`)
   - Complete error message
   - Steps to reproduce

## 🎓 Understanding the Code

### Key Components

**`src/benchmark.py`**
- Main training and evaluation logic
- Dataset loading and preprocessing
- Model training loop
- Evaluation metrics and visualization

**`run_custom_experiment.py`**
- Example of running custom experiments
- Shows how to adjust sampling weights
- Demonstrates parameter customization

**`test_fear_sadness_experiment.py`**
- Example of focusing on specific emotions
- Uses higher sampling weights for fear and sadness
- Runs longer training (100 epochs)

### Modifying Experiments

You can create your own experiment by copying `run_custom_experiment.py` and modifying:

```python
config = create_experiment_config(
    experiment_name="my_custom_experiment",
    sampling_weights={
        'angry': 1.0,
        'disgust': 1.0,
        'fear': 2.0,      # Increase weight for fear
        'happy': 1.0,
        'neutral': 1.0,
        'sad': 2.0,       # Increase weight for sad
        'surprise': 1.0
    },
    num_epochs=10,        # Number of training epochs
    sample_per_class=100  # Samples per emotion class
)
```

## 🧪 Verification Steps

After installation, verify everything is set up correctly.

### Automated Verification (Recommended)

We provide a verification script that checks everything automatically:

```bash
source .virtual_environment/bin/activate
python verify_setup.py
```

This script will check:
- ✅ Python version
- ✅ Virtual environment activation
- ✅ All required packages
- ✅ Project file structure
- ✅ Dataset presence and structure

**Expected output:**
```
🔍 CS6350 Emotion ML - Setup Verification
============================================================
✅ Python Version                : PASS
✅ Virtual Environment           : PASS
✅ Required Packages             : PASS
✅ Project Files                 : PASS
✅ Dataset                       : PASS
============================================================
🎉 All checks passed! Your environment is ready.
```

### Manual Verification (Optional)

If you prefer to verify manually:

#### 1. Check Python Installation
```bash
python3 --version
# Should show: Python 3.12.x
```

#### 2. Check Virtual Environment
```bash
source .virtual_environment/bin/activate
which python
# Should show path to .virtual_environment/bin/python
```

#### 3. Check Installed Packages
```bash
source .virtual_environment/bin/activate
pip list | grep torch
# Should show torch and torchvision packages
```

#### 4. Check Dataset
```bash
ls data/train/
# Should show 7 emotion directories
ls data/train/happy/ | wc -l
# Should show number of happy images
```

#### 5. Test Import
```bash
source .virtual_environment/bin/activate
python -c "import torch; import transformers; print('✓ All imports successful')"
# Should print: ✓ All imports successful
```

## 🚀 Next Steps

Once you have successfully run the benchmark:

1. **Explore the results** in `experiments/` directory
2. **Modify experiments** to focus on specific emotions
3. **Adjust hyperparameters** (epochs, batch size, learning rate)
4. **Try different sampling weights** to improve specific emotion detection
5. **Review the project roadmap** in `PROJECT_ROADMAP.md` for future development tasks

## 📚 Additional Resources

- **FER-2013 Dataset**: https://www.kaggle.com/datasets/msambare/fer2013
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Transformers Library**: https://huggingface.co/docs/transformers/
- **Pre-trained Model**: https://huggingface.co/dima806/facial_emotions_image_detection

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

**Need help?** Open an issue on GitHub or refer to the troubleshooting section above.
