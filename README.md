# cs6350-emotion-ml
Detecting change in emotion given a time series of images

## 🚀 Quick Start

### Running Your First Test
```bash
# Quick start - run basic experiment
python scripts/quick_start.py --experiment basic

# Run all experiment types
python scripts/quick_start.py --experiment all --epochs 3 --samples 50

# Run focused experiments
python scripts/run_custom_experiment.py
python scripts/test_fear_sadness_experiment.py

# Test the enhanced checkpoint system
python scripts/test_enhanced_checkpoints.py
```

### Continuing Training with Ratio Manipulation
```bash
# List available checkpoints
python scripts/resume_experiment.py --list

# Resume from a checkpoint with same settings
python scripts/resume_experiment.py \
    --checkpoint "./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt" \
    --name "resumed_experiment" \
    --epochs 5

# Resume with different sampling weights (ratio manipulation)
python scripts/resume_experiment.py \
    --checkpoint "./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt" \
    --name "different_weights_experiment" \
    --epochs 5 \
    --weights '{"disgust": 5.0, "fear": 3.0, "happy": 0.5}' \
    --description "Resumed with different sampling weights"
```

### Project Structure
```
cs6350-emotion-ml/
├── src/                    # Core ML code
│   └── benchmark.py        # Enhanced benchmark with checkpointing
├── scripts/               # Training and experiment scripts
│   ├── quick_start.py           # Easy experiment runner
│   ├── run_custom_experiment.py
│   ├── test_fear_sadness_experiment.py
│   ├── test_enhanced_checkpoints.py
│   └── resume_experiment.py
├── docs/                  # Documentation
│   └── CHECKPOINT_SYSTEM.md
├── experiments/           # Experiment outputs (auto-created)
├── data/                 # Dataset (train/test splits)
└── requirements.txt      # Dependencies
```

## 📊 First Run - Benchmark Results

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

## 🔄 Enhanced Checkpoint System

This project includes an advanced checkpoint system that allows you to:

### ✅ **Complete Training State Preservation**
- Save model weights, optimizer state, epoch number, best accuracy, and training history
- Resume training from any saved checkpoint
- Never lose progress due to interruptions

### 📊 **Training Ratio Manipulation**
- Change sampling weights between experiments
- Track how different data ratios affect model performance
- Maintain experiment lineage and change logs

### 🎯 **Key Features**
- **Experiment Isolation**: Each experiment gets its own timestamped folder
- **Ratio Tracking**: Logs when sampling weights change between experiments
- **Resume Capability**: Continue from any checkpoint with same or different configuration
- **Backward Compatibility**: Works with existing model files

### 📁 **Experiment Structure**
Each experiment creates:
```
experiments/experiment_name_20250108_143022/
├── experiment_config.json          # Configuration
├── experiment_changes.log           # Ratio change tracking
├── checkpoint_epoch_1.pt           # Complete state checkpoints
├── best_checkpoint.pt              # Best performing model
├── classification_report.txt       # Results
├── confusion_matrix_benchmark.png  # Visualizations
└── training_history.csv            # Training data
```

### 🛠️ **Usage Examples**

**Basic Experiment:**
```python
from src.benchmark import create_experiment_config, run_experiment

config = create_experiment_config(
    experiment_name="my_experiment",
    sampling_weights={
        'disgust': 3.0,  # Higher weight for underrepresented class
        'fear': 2.0,     # Higher weight for underrepresented class
        'happy': 0.8,    # Lower weight for overrepresented class
    },
    num_epochs=10,
    experiment_description="Testing different sampling weights"
)

output_dir, best_acc = run_experiment(config)
```

**Resume with Different Ratios:**
```python
config = create_experiment_config(
    experiment_name="different_weights_experiment",
    sampling_weights={
        'disgust': 5.0,  # Even higher weight
        'fear': 3.0,     # Even higher weight
        'happy': 0.5,    # Even lower weight
    },
    resume_from_checkpoint="./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt",
    num_epochs=5,
    experiment_description="Resumed with different sampling weights"
)
```

For detailed documentation, see [docs/CHECKPOINT_SYSTEM.md](docs/CHECKPOINT_SYSTEM.md).

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
