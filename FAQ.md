# Frequently Asked Questions (FAQ) ❓

## General Questions

### What is this project about?

This project uses machine learning to detect emotions from facial expressions in images. It can identify seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

### What dataset does this project use?

We use the FER-2013 (Facial Expression Recognition 2013) dataset from Kaggle, which contains thousands of grayscale facial images labeled with emotions.

### Do I need prior machine learning experience?

No! This project is designed to be accessible to beginners. Follow the setup instructions in the README, and you'll be able to run experiments without deep ML knowledge. However, some Python familiarity is helpful.

### What are the system requirements?

- **Python**: 3.12 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space for dataset
- **GPU**: Optional but significantly speeds up training
- **OS**: Linux (recommended), macOS, or Windows (with WSL)

---

## Setup Questions

### Why do I need Python 3.12 specifically?

The project was developed and tested with Python 3.12. While it may work with other versions (3.10+), Python 3.12 ensures compatibility with all dependencies.

### Can I use a different Python version?

You can try, but you may encounter compatibility issues. We recommend using Python 3.12 as specified. Use `pyenv` or `conda` to manage multiple Python versions if needed.

### What is a virtual environment and why do I need it?

A virtual environment is an isolated Python environment that keeps project dependencies separate from your system Python. This prevents conflicts between different projects and makes the project portable.

### How do I know if I'm in the virtual environment?

When activated, your terminal prompt will show `(.virtual_environment)` at the beginning. You can also run:
```bash
which python
# Should show: /path/to/cs6350-emotion-ml/.virtual_environment/bin/python
```

### The dataset download is very slow. Any tips?

The FER-2013 dataset is about 300MB compressed. Tips:
- Use a stable internet connection
- Download during off-peak hours
- Consider using Kaggle CLI for resumable downloads
- Be patient—it's a one-time download

---

## Running Questions

### How long does the benchmark take to run?

- **With GPU**: 15-30 minutes
- **Without GPU**: 1-3 hours

This depends on your hardware. The default configuration uses 100 samples per class and 5 epochs.

### Can I make training faster?

Yes! For testing purposes, you can:
```python
config = create_experiment_config(
    experiment_name="quick_test",
    num_epochs=1,           # Reduce from 5
    sample_per_class=20     # Reduce from 100
)
```

This will run much faster but with lower accuracy.

### What if I don't have a GPU?

The project works fine without a GPU—it will just take longer. PyTorch automatically detects and uses available GPUs. You can check GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Where are the results saved?

Results are saved in `experiments/[experiment_name]_[timestamp]/` with:
- `best_model.pt` - Trained model
- `classification_report.txt` - Detailed metrics
- `confusion_matrix_benchmark.png` - Visual confusion matrix
- `training_history.png` - Loss and accuracy plots
- `training_history.csv` - Training data

---

## Error Questions

### ImportError: No module named 'torch'

**Solution**: Activate the virtual environment first:
```bash
source .virtual_environment/bin/activate
```

If still failing:
```bash
pip install --upgrade -r requirements.txt
```

### FileNotFoundError: data/train

**Solution**: Download the dataset from Kaggle:
1. Visit https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract
3. Copy `train/` and `test/` folders to `data/`

See the README's "Step 4: Download the Dataset" for details.

### CUDA out of memory

**Solutions**:
1. Reduce batch size: `batch_size=16` (default is 32)
2. Reduce samples: `sample_per_class=50` (default is 100)
3. Close other GPU-using applications
4. Use CPU instead (slower but works)

### make: command not found

**Linux/macOS**: Install make:
```bash
# Linux
sudo apt install build-essential

# macOS
xcode-select --install
```

**Windows**: Use commands directly instead of make, or use WSL.

### Permission denied when installing

**Solution**: On Linux, system packages need sudo:
```bash
sudo apt install python3.12-venv ffmpeg
```

Python packages should NOT need sudo (they install in virtual environment).

---

## Usage Questions

### How do I create my own experiment?

1. Copy an example:
   ```bash
   cp run_custom_experiment.py my_experiment.py
   ```

2. Edit the configuration:
   ```python
   config = create_experiment_config(
       experiment_name="my_test",
       sampling_weights={'fear': 2.0, 'sad': 2.0},  # Focus on these
       num_epochs=5
   )
   ```

3. Run it:
   ```bash
   python my_experiment.py
   ```

### What are sampling weights?

Sampling weights control how often each emotion appears during training. Higher weights mean the model sees that emotion more often.

Example:
```python
sampling_weights={
    'fear': 2.0,  # 2x more frequent
    'happy': 1.0, # Normal frequency
    'sad': 0.5    # Half as frequent
}
```

### How do I interpret the results?

Key metrics:
- **Accuracy**: % of correct predictions (higher is better)
- **Precision**: Of predicted emotion X, % that were actually X
- **Recall**: Of actual emotion X, % that were detected
- **F1-Score**: Balance between precision and recall

Example: If "happy" has 80% precision and 75% recall:
- 80% of "happy" predictions were correct
- 75% of actual happy faces were detected

### Can I use my own images?

Yes, but you'll need to organize them like the FER-2013 dataset:
```
my_data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

Then modify `TRAIN_DIR` and `TEST_DIR` in `src/benchmark.py`.

---

## Customization Questions

### How do I change the number of epochs?

In your experiment script:
```python
config = create_experiment_config(
    experiment_name="longer_training",
    num_epochs=20  # Change this
)
```

More epochs = longer training but potentially better results.

### How do I change the learning rate?

```python
config = create_experiment_config(
    experiment_name="different_lr",
    learning_rate=1e-5  # Lower = slower but more stable
)
```

### Can I use a different model?

Yes! Modify the model name in the configuration:
```python
config = create_experiment_config(
    experiment_name="different_model",
    model_name="your-huggingface-model-name"
)
```

Note: The model must be compatible with image classification tasks.

### How do I focus on specific emotions?

Use sampling weights:
```python
config = create_experiment_config(
    experiment_name="fear_focus",
    sampling_weights={
        'angry': 1.0,
        'disgust': 1.0,
        'fear': 3.0,    # 3x focus
        'happy': 1.0,
        'neutral': 1.0,
        'sad': 1.0,
        'surprise': 1.0
    }
)
```

---

## Contribution Questions

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. You can:
- Report bugs
- Suggest features
- Improve documentation
- Submit code improvements

### I found a bug. What should I do?

1. Check if it's already reported in GitHub Issues
2. If not, create a new issue with:
   - Clear description
   - Steps to reproduce
   - Your environment (OS, Python version)
   - Error messages

### I have an idea for improvement. Where do I suggest it?

Create a GitHub Issue with:
- Clear description of the feature
- Use case/benefits
- Possible implementation approach (if you have ideas)

---

## Performance Questions

### Why is the accuracy only 61%?

This is the baseline performance. Emotions are subjective and challenging even for humans. The model:
- Excels at distinct emotions (disgust, happy, surprise): 75-95%
- Struggles with subtle emotions (fear, sad, angry): 40-54%

This is significantly better than random guessing (14.3% for 7 classes).

### How can I improve the accuracy?

Try:
1. Train longer (more epochs)
2. Use more training data
3. Adjust sampling weights for problematic emotions
4. Try different models
5. Add data augmentation
6. Fine-tune hyperparameters

### Which emotions are easiest/hardest to detect?

**Easiest** (from benchmark):
1. Disgust: 96% recall
2. Happy: 77% recall
3. Surprise: 78% recall

**Hardest**:
1. Fear: 24% recall
2. Sad: 41% recall
3. Angry: 54% recall

---

## Platform-Specific Questions

### Does this work on Windows?

The project is primarily tested on Linux/macOS. For Windows:
- Use WSL (Windows Subsystem for Linux) - Recommended
- Or use Git Bash and adapt commands
- PowerShell may require syntax modifications

### Does this work on Apple Silicon (M1/M2)?

Yes, but:
- PyTorch supports Apple Silicon
- May need to install different PyTorch version
- Performance is good but different from CUDA GPU

### Can I run this in Google Colab?

Yes! Upload the files and:
```python
!pip install -r requirements.txt
# Upload dataset or use Kaggle API
# Run experiments
```

Colab provides free GPU access which speeds up training.

---

## Data Questions

### Is the FER-2013 dataset free?

Yes, it's freely available on Kaggle for research and educational purposes.

### Do I need a Kaggle account?

Yes, you need a Kaggle account to download datasets. It's free to create.

### Can I use a different dataset?

Yes! Organize it in the same structure (emotion subfolders) and update the paths in the code.

### How many images are in the dataset?

Approximately:
- **Training**: 28,000 images
- **Testing**: 7,000 images
- Distributed across 7 emotion classes

---

## Miscellaneous Questions

### What's the difference between train and test data?

- **Training data**: Used to teach the model
- **Test data**: Used to evaluate performance on unseen data

Never train on test data—it would give misleading performance metrics.

### What does "epoch" mean?

One epoch = one complete pass through all training data. More epochs generally improve performance up to a point (then overfitting can occur).

### What is a confusion matrix?

A visualization showing:
- Rows: Actual emotions
- Columns: Predicted emotions
- Values: Number of predictions

Diagonal = correct predictions. Off-diagonal = confusions.

### Why are some emotions confused more than others?

Similar emotions are easily confused:
- Fear ↔ Surprise (both have wide eyes)
- Angry ↔ Disgust (similar facial features)
- Sad ↔ Neutral (subtle differences)

This is normal and even humans can struggle with these distinctions.

---

## Still Have Questions?

1. Check the [README.md](README.md) for detailed documentation
2. Review [WORKFLOW.md](WORKFLOW.md) for process guides
3. Search existing GitHub Issues
4. Create a new GitHub Issue with your question

We're here to help! 🤝
