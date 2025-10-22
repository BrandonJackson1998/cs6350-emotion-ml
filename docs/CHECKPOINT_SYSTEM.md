# Enhanced Checkpoint System

This document describes the enhanced checkpoint system for emotion recognition experiments, which provides complete training state preservation, experiment isolation, and training data ratio tracking.

## Features

### 🔄 Complete Checkpoint System
- **Full State Preservation**: Saves model weights, optimizer state, epoch number, best accuracy, and training history
- **Resume Capability**: Continue training from any saved checkpoint
- **Best Model Tracking**: Automatically saves the best performing model
- **Backward Compatibility**: Maintains compatibility with existing model files

### 📁 Experiment Isolation
- **Unique Folders**: Each experiment gets its own timestamped directory
- **Complete Metadata**: Saves experiment configuration, description, and timestamps
- **Isolated Logs**: Each experiment maintains its own change log

### 📊 Training Data Ratio Tracking
- **Change Logging**: Tracks when sampling weights change between experiments
- **Ratio History**: Maintains history of training data distributions
- **Experiment Lineage**: Shows how experiments relate to each other

## Usage

### Basic Experiment
```python
from src.benchmark import create_experiment_config, run_experiment

config = create_experiment_config(
    experiment_name="my_experiment",
    sampling_weights={
        'angry': 1.0,
        'disgust': 3.0,  # Higher weight for underrepresented class
        'fear': 2.0,
        'happy': 0.8,    # Lower weight for overrepresented class
        'neutral': 1.0,
        'sad': 1.0,
        'surprise': 1.5
    },
    num_epochs=10,
    sample_per_class=100,
    experiment_description="Testing different sampling weights"
)

output_dir, best_acc = run_experiment(config)
```

### Resume from Checkpoint
```python
# Resume with same configuration
config = create_experiment_config(
    experiment_name="resumed_experiment",
    resume_from_checkpoint="./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt",
    num_epochs=5,  # Additional epochs
    experiment_description="Resumed from epoch 5"
)

output_dir, best_acc = run_experiment(config)
```

### Resume with Different Sampling Weights
```python
# Resume with different sampling weights
config = create_experiment_config(
    experiment_name="different_weights_experiment",
    sampling_weights={
        'angry': 1.0,
        'disgust': 5.0,  # Even higher weight
        'fear': 3.0,
        'happy': 0.5,    # Even lower weight
        'neutral': 1.0,
        'sad': 1.0,
        'surprise': 2.0
    },
    resume_from_checkpoint="./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt",
    num_epochs=5,
    experiment_description="Resumed with different sampling weights"
)

output_dir, best_acc = run_experiment(config)
```

## Command Line Tools

### List Available Checkpoints
```bash
python resume_experiment.py --list
```

### Resume from Checkpoint
```bash
python resume_experiment.py \
    --checkpoint "./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt" \
    --name "resumed_experiment" \
    --epochs 5 \
    --description "Resumed training"
```

### Resume with Different Sampling Weights
```bash
python resume_experiment.py \
    --checkpoint "./experiments/my_experiment_20250108_143022/checkpoint_epoch_5.pt" \
    --name "different_weights_experiment" \
    --epochs 5 \
    --weights '{"disgust": 5.0, "fear": 3.0, "happy": 0.5}' \
    --description "Resumed with different sampling weights"
```

## File Structure

Each experiment creates a directory with the following structure:

```
experiments/
└── experiment_name_20250108_143022/
    ├── experiment_config.json          # Experiment configuration
    ├── experiment_changes.log           # Change log with ratio tracking
    ├── checkpoint_epoch_1.pt           # Checkpoint for each epoch
    ├── checkpoint_epoch_2.pt
    ├── ...
    ├── best_checkpoint.pt              # Best performing checkpoint
    ├── epoch_1_model.pt                # Individual epoch models (compatibility)
    ├── epoch_2_model.pt
    ├── ...
    ├── classification_report.txt       # Final classification report
    ├── confusion_matrix_benchmark.png  # Confusion matrix plot
    ├── training_history.png            # Training curves
    └── training_history.csv            # Training data
```

## Checkpoint Contents

Each checkpoint file contains:
- `epoch`: Current epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (AdamW)
- `best_acc`: Best validation accuracy so far
- `history`: Complete training history (loss, accuracy)
- `config`: Experiment configuration
- `timestamp`: When checkpoint was saved

## Experiment Logging

The system automatically logs:
- **Experiment Start**: Timestamp, description, initial configuration
- **Resume Events**: When resuming from checkpoints
- **Ratio Changes**: When sampling weights change between experiments
- **Configuration Changes**: Any modifications to experiment parameters

## Testing the System

Run the test script to see the enhanced checkpoint system in action:

```bash
python test_enhanced_checkpoints.py
```

This will:
1. Run an initial 3-epoch experiment
2. Resume with the same configuration for 2 more epochs
3. Resume with different sampling weights for 2 more epochs
4. Show the experiment logs and change tracking

## Benefits

1. **No Lost Work**: Never lose training progress due to interruptions
2. **Flexible Experimentation**: Easily test different sampling strategies
3. **Reproducible Results**: Complete state preservation ensures reproducibility
4. **Experiment Tracking**: Clear lineage of how experiments relate to each other
5. **Resource Efficiency**: Resume from any point without starting over
6. **Data Ratio Awareness**: Track how sampling changes affect model performance

## Migration from Old System

The enhanced system is backward compatible:
- Existing model files (`.pt`) still work
- Old experiments can be resumed using the new system
- Gradual migration is supported

## Best Practices

1. **Regular Checkpoints**: The system saves checkpoints every epoch automatically
2. **Descriptive Names**: Use clear experiment names and descriptions
3. **Log Changes**: Document why you're changing sampling weights
4. **Test Incrementally**: Start with small experiments to test configurations
5. **Monitor Logs**: Check `experiment_changes.log` to track experiment lineage
