# Development Workflow Guide 🔄

This document explains the typical workflows for working with this project.

## First-Time Setup Workflow

```
1. Clone Repository
   ↓
2. Install System Dependencies
   ↓
3. Create Virtual Environment & Install Python Packages
   ↓
4. Download Dataset from Kaggle
   ↓
5. Verify Setup
   ↓
6. Run Benchmark
   ↓
7. Analyze Results
```

### Detailed Steps

#### 1. Clone Repository
```bash
git clone https://github.com/BrandonJackson1998/cs6350-emotion-ml.git
cd cs6350-emotion-ml
```

#### 2. Install System Dependencies
```bash
# Linux
make install-deb

# macOS
make install-deb-mac
```

#### 3. Install Python Packages
```bash
make install-pip
```

#### 4. Download Dataset
- Visit: https://www.kaggle.com/datasets/msambare/fer2013
- Download and extract to `data/train/` and `data/test/`

#### 5. Verify Setup
```bash
make verify
```

#### 6. Run Benchmark
```bash
make benchmark
```

#### 7. Analyze Results
Check `experiments/` directory for outputs

---

## Daily Development Workflow

```
1. Activate Virtual Environment
   ↓
2. Make Code Changes
   ↓
3. Test Changes
   ↓
4. Run Experiments
   ↓
5. Analyze Results
   ↓
6. Commit Changes
```

### Detailed Steps

#### 1. Activate Virtual Environment
```bash
source .virtual_environment/bin/activate
```

#### 2. Make Code Changes
Edit files in `src/` or create new experiment scripts

#### 3. Test Changes
```bash
# Quick syntax check
python -m py_compile src/benchmark.py

# Test imports
python -c "from src.benchmark import create_experiment_config"
```

#### 4. Run Experiments
```bash
# Quick test with fewer samples
python run_custom_experiment.py

# Or modify config for faster testing
# (e.g., num_epochs=1, sample_per_class=10)
```

#### 5. Analyze Results
```bash
# Check outputs
ls experiments/

# View classification report
cat experiments/latest_experiment/classification_report.txt

# View confusion matrix
open experiments/latest_experiment/confusion_matrix_benchmark.png
```

#### 6. Commit Changes
```bash
git add .
git commit -m "Description of changes"
git push
```

---

## Experiment Customization Workflow

### Creating a Custom Experiment

1. **Copy Template**
   ```bash
   cp run_custom_experiment.py my_experiment.py
   ```

2. **Modify Configuration**
   ```python
   config = create_experiment_config(
       experiment_name="my_custom_experiment",
       sampling_weights={
           'angry': 1.0,
           'disgust': 2.0,    # Adjust weights
           'fear': 1.5,
           'happy': 1.0,
           'neutral': 1.0,
           'sad': 1.5,
           'surprise': 1.0
       },
       num_epochs=10,         # Adjust epochs
       sample_per_class=100   # Adjust samples
   )
   ```

3. **Run Experiment**
   ```bash
   source .virtual_environment/bin/activate
   python my_experiment.py
   ```

4. **Compare Results**
   - Check accuracy in terminal output
   - Review classification reports
   - Compare confusion matrices
   - Analyze training history plots

---

## Troubleshooting Workflow

### Issue: Virtual Environment Problems

```
Problem Detected
   ↓
Verify virtual environment exists
   ↓
   NO → Run: make install-pip
   ↓
   YES → Activate it
   ↓
Verify packages installed
   ↓
   NO → Run: pip install -r requirements.txt
   ↓
   YES → Run: make verify
```

### Issue: Dataset Problems

```
Problem Detected
   ↓
Check if data/train/ exists
   ↓
   NO → Download from Kaggle
   ↓
   YES → Check emotion subdirectories
   ↓
   Missing? → Re-extract dataset
   ↓
   Present? → Run: make verify
```

### Issue: Import Errors

```
ImportError Occurs
   ↓
Activate virtual environment
   ↓
Check package installed: pip list | grep [package]
   ↓
   Not Found → Install: pip install [package]
   ↓
   Found → Check Python version: python --version
   ↓
   Wrong version? → Use Python 3.12
   ↓
   Correct version? → Reinstall package
```

---

## Data Science Workflow

### Hypothesis → Experiment → Analysis

1. **Form Hypothesis**
   - Example: "Increasing sampling weight for 'fear' will improve its detection"

2. **Design Experiment**
   ```python
   config = create_experiment_config(
       experiment_name="fear_focus",
       sampling_weights={
           'fear': 3.0,  # Triple weight for fear
           # ... other emotions at 1.0
       }
   )
   ```

3. **Run Experiment**
   ```bash
   python my_fear_experiment.py
   ```

4. **Analyze Results**
   - Compare 'fear' metrics with baseline
   - Check if recall/precision improved
   - Look at confusion matrix
   - Review training history

5. **Document Findings**
   - Update experiment notes
   - Save best configurations
   - Plan next experiment

---

## Performance Optimization Workflow

### Making Training Faster

1. **Reduce Dataset Size** (for testing)
   ```python
   sample_per_class=50  # Instead of 100
   ```

2. **Reduce Epochs** (for quick tests)
   ```python
   num_epochs=3  # Instead of 5 or more
   ```

3. **Adjust Batch Size**
   ```python
   batch_size=64  # Larger = faster but more memory
   ```

4. **Use GPU** (if available)
   - PyTorch automatically uses CUDA if available
   - Check with: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Collaboration Workflow

### Working with Others

1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/my-improvement
   ```

3. **Make Changes & Test**
   ```bash
   # Make changes
   make verify
   python my_experiment.py
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: description"
   git push origin feature/my-improvement
   ```

5. **Create Pull Request**
   - Go to GitHub
   - Create PR from your branch
   - Add description and results

6. **Review and Merge**
   - Address feedback
   - Merge when approved

---

## Best Practices

### ✅ Do's

- Always activate virtual environment before working
- Run `make verify` after setup changes
- Use descriptive experiment names
- Document your changes
- Commit working code
- Save experiment configurations

### ❌ Don'ts

- Don't commit the virtual environment (`.virtual_environment/`)
- Don't commit large model files unless necessary
- Don't modify data files
- Don't hardcode paths
- Don't skip verification steps
- Don't run long experiments without testing first

---

## Quick Reference

### Most Common Commands

```bash
# Setup
make install
make verify

# Daily use
source .virtual_environment/bin/activate
make benchmark
python my_experiment.py

# Checking
make verify
ls experiments/
cat experiments/*/classification_report.txt

# Cleanup
deactivate  # Exit virtual environment
```

### File Organization

```
experiments/
└── [experiment_name]_[timestamp]/
    ├── best_model.pt                 # Use for inference
    ├── classification_report.txt     # Check metrics
    ├── confusion_matrix_benchmark.png # Visualize confusion
    └── training_history.png          # Track learning
```

---

For more information, see:
- [README.md](README.md) - Complete documentation
- [QUICKSTART.md](QUICKSTART.md) - Fast setup guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Future plans
