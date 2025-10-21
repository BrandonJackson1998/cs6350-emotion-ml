# cs6350-emotion-ml
Detecting change in emotion given a time series of images

## Installation & Setup

This project uses Python 3.12, PyTorch, Transformers, and requires ffmpeg for audio processing capabilities.

### Prerequisites

All platforms require:
- **Python 3.12** (or compatible version)
- **ffmpeg** (for audio file processing)
- **Git** (to clone the repository)

### Installation Instructions

#### Linux (Ubuntu/Debian)

The project includes a Makefile for easy installation on Linux systems with apt package management:

```bash
# Clone the repository
git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
cd cs6350-emotion-ml

# Install system dependencies and Python packages
make install
```

This will:
1. Install `python3.12-venv` and `ffmpeg` using apt
2. Create a virtual environment at `.virtual_environment`
3. Install all Python dependencies from `requirements.txt`

#### macOS

For macOS systems with Homebrew:

```bash
# Clone the repository
git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
cd cs6350-emotion-ml

# Install system dependencies and Python packages
make install-mac
```

This will:
1. Install `python@3.12` and `ffmpeg` using Homebrew
2. Create a virtual environment at `.virtual_environment`
3. Install all Python dependencies from `requirements.txt`

#### Windows

For Windows systems, follow these manual installation steps:

1. **Install Python 3.12:**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install ffmpeg:**
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) or use a package manager like [Chocolatey](https://chocolatey.org/):
     ```powershell
     # Using Chocolatey (run PowerShell as Administrator)
     choco install ffmpeg
     ```
   - Or manually:
     - Download the build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
     - Extract to a folder (e.g., `C:\ffmpeg`)
     - Add `C:\ffmpeg\bin` to your System PATH

3. **Clone the repository:**
   ```powershell
   git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
   cd cs6350-emotion-ml
   ```

4. **Create a virtual environment:**
   ```powershell
   python -m venv .virtual_environment
   ```

5. **Activate the virtual environment:**
   ```powershell
   # PowerShell
   .\.virtual_environment\Scripts\Activate.ps1
   
   # Command Prompt
   .\.virtual_environment\Scripts\activate.bat
   ```

6. **Install Python dependencies:**
   ```powershell
   pip install --upgrade -r requirements.txt
   ```

### Verify Installation

After installation, activate your virtual environment and verify the setup:

```bash
# Linux/macOS
source .virtual_environment/bin/activate

# Windows PowerShell
.\.virtual_environment\Scripts\Activate.ps1

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Running the Project

Once installed, you can run experiments:

```bash
# Activate virtual environment first (if not already activated)
# Linux/macOS: source .virtual_environment/bin/activate
# Windows: .\.virtual_environment\Scripts\Activate.ps1

# Run the benchmark
python -m src.benchmark

# Run custom experiments
python run_custom_experiment.py
python test_fear_sadness_experiment.py
```

### Dataset Setup

This project uses the FER-2013 dataset. You'll need to download it from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and organize it in the following structure:

```
cs6350-emotion-ml/
├── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
```

## First Run - Benchmark Results

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



## Project TODO List

### 1. Problem Definition & Goals
- [ ] Define emotion transition detection for educational monitoring
  - Detect transitions between 7 emotional states: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - Focus on monitoring students' emotional states during learning sessions
- [ ] Establish success criteria and evaluation metrics
  - Primary metric: Accuracy of emotion transition detection
  - Secondary metric: Processing speed for real-time application
- [ ] Define target emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- [ ] Set performance benchmarks
  - Target accuracy: >85% for emotion classification
  - Target speed: Real-time processing capability (<100ms per frame)
  - Application: Educational technology for student engagement monitoring

### 2. Data Collection
- [ ] Download FER-2013 dataset from Kaggle
  - Source: https://www.kaggle.com/datasets/msambare/fer2013/data?select=test
  - Target: ~30,000 images for training, ~10,000 for testing
- [ ] Verify dataset contains all required emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- [ ] Document dataset characteristics and limitations
- [ ] Set up data storage structure and organization
- [ ] Validate data integrity and completeness

### 3. Data Preparation
- [ ] Clean and validate image data
- [ ] Preprocess images (resize, normalize, etc.)
- [ ] Handle missing or corrupted data
- [ ] Create train/validation/test splits
- [ ] Format data for model consumption

### 4. Feature Engineering
- [ ] Extract facial features from images
- [ ] Create temporal features from time series
- [ ] Select most informative features
- [ ] Handle feature scaling and normalization
- [ ] Create feature pipelines

### 5. Model Training
- [ ] Select appropriate ML algorithms
- [ ] Implement baseline models
- [ ] Train deep learning models (CNNs, RNNs, etc.)
- [ ] Handle time series aspects of the data
- [ ] Cross-validation setup

### 6. Evaluation
- [ ] Implement evaluation metrics
- [ ] Create test harness
- [ ] Perform model validation
- [ ] Compare model performances
- [ ] Generate evaluation reports

### 7. Hyperparameter Tuning
- [ ] Define hyperparameter search space
- [ ] Implement tuning strategy (grid search, random search, etc.)
- [ ] Optimize model parameters
- [ ] Document best parameters found

### 8. Deployment
- [ ] Package model for deployment
- [ ] Create inference pipeline
- [ ] Build API or application interface
- [ ] Test deployment setup
- [ ] Create documentation for usage

### 9. Monitoring & Maintenance
- [ ] Implement performance monitoring
- [ ] Set up model drift detection
- [ ] Create update/retraining pipeline
- [ ] Plan for model versioning
- [ ] Document maintenance procedures
