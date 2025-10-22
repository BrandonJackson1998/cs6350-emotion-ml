# benchmark.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import signal
import sys
import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for interrupt handling
TRAINING_INTERRUPTED = False
CURRENT_CHECKPOINT_INFO = None

# Data paths - relative to project root
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Emotion labels (ordered to match pre-trained model expectations)
# Model expects: sad, disgust, angry, neutral, fear, surprise, happy
EMOTION_LABELS = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}

class FER2013FolderDataset(Dataset):
    """Custom Dataset for FER2013 organized in folders"""
    
    def __init__(self, root_dir, processor, emotion_labels=EMOTION_LABELS, max_samples_per_class=None, 
                 sampling_weights=None, use_full_dataset=False):
        self.root_dir = root_dir
        self.processor = processor
        self.emotion_labels = emotion_labels
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
        self.sampling_weights = sampling_weights
        self.use_full_dataset = use_full_dataset
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_counts = {}
        
        for emotion in emotion_labels:
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Use full dataset or limit samples per class
            if use_full_dataset:
                # Use all available images
                selected_files = image_files
                print(f"Using full dataset for {emotion}: {len(selected_files)} images")
            elif max_samples_per_class:
                # Limit samples per class if specified (for benchmarking)
                selected_files = image_files[:max_samples_per_class]
                print(f"Limited dataset for {emotion}: {len(selected_files)}/{len(image_files)} images")
            else:
                # Use all available if no limit specified
                selected_files = image_files
            
            self.class_counts[emotion] = len(selected_files)
            
            for img_file in selected_files:
                self.image_paths.append(os.path.join(emotion_dir, img_file))
                self.labels.append(self.emotion_to_idx[emotion])
        
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
        print("Class distribution:")
        for emotion, count in self.class_counts.items():
            print(f"  {emotion:10s}: {count:5d} images")
        
        if self.sampling_weights:
            print(f"Using sampling weights: {self.sampling_weights}")
    
    def get_sampling_weights(self):
        """Get sampling weights for WeightedRandomSampler"""
        if not self.sampling_weights:
            return None
        
        weights = []
        for label in self.labels:
            emotion = self.emotion_labels[label]
            weight = self.sampling_weights.get(emotion, 1.0)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (48, 48), color='black')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_dataset_statistics(root_dir, emotion_labels=EMOTION_LABELS):
    """Get statistics about the dataset"""
    stats = {}
    total = 0
    
    for emotion in emotion_labels:
        emotion_dir = os.path.join(root_dir, emotion)
        if os.path.exists(emotion_dir):
            count = len([f for f in os.listdir(emotion_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[emotion] = count
            total += count
        else:
            stats[emotion] = 0
    
    return stats, total

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    global TRAINING_INTERRUPTED, CURRENT_CHECKPOINT_INFO
    print(f"\n⚠️  Training interrupted by signal {signum}")
    print("Saving checkpoint before exit...")
    TRAINING_INTERRUPTED = True

def setup_signal_handlers():
    """Setup signal handlers for graceful interruption"""
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    print("✓ Signal handlers registered for graceful interruption")

def train_epoch(model, dataloader, optimizer, criterion, device, config=None, 
                output_dir=None, epoch=0, best_acc=0, history=None):
    """Train for one epoch with batch-level checkpointing"""
    global TRAINING_INTERRUPTED
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    batch_checkpoint_frequency = config.get('batch_checkpoint_frequency', 100) if config else 100
    total_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, batch in enumerate(progress_bar):
        # Check for interruption
        if TRAINING_INTERRUPTED:
            print("\n⚠️  Training interrupted, saving checkpoint...")
            if config and output_dir and history is not None:
                save_checkpoint(model, optimizer, epoch, best_acc, history, config, 
                              output_dir, current_batch=batch_idx, total_batches=total_batches)
            break
            
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress with memory usage
        memory_info = get_memory_usage()
        progress_bar.set_postfix({
            'loss': loss.item(), 
            'acc': correct/total,
            'gpu_mem': f"{memory_info['allocated']:.1f}GB"
        })
        
        # Save batch checkpoint periodically
        if (config and output_dir and history is not None and 
            batch_checkpoint_frequency > 0 and 
            (batch_idx + 1) % batch_checkpoint_frequency == 0):
            
            print(f"\n💾 Saving batch checkpoint at batch {batch_idx + 1}/{total_batches}")
            save_checkpoint(model, optimizer, epoch, best_acc, history, config, 
                          output_dir, current_batch=batch_idx, total_batches=total_batches)
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            
            predictions = outputs.logits.argmax(dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels, total_loss / len(dataloader)

def create_experiment_config(experiment_name, sampling_weights=None, **kwargs):
    """Create experiment configuration"""
    config = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'sampling_weights': sampling_weights,
        'model_name': kwargs.get('model_name', "dima806/facial_emotions_image_detection"),
        'batch_size': kwargs.get('batch_size', 32),
        'learning_rate': kwargs.get('learning_rate', 2e-5),
        'num_epochs': kwargs.get('num_epochs', 5),
        'sample_per_class': kwargs.get('sample_per_class', 100),
        'use_full_dataset': kwargs.get('use_full_dataset', False),
        'batch_checkpoint_frequency': kwargs.get('batch_checkpoint_frequency', 100),
        'auto_checkpoint_on_interrupt': kwargs.get('auto_checkpoint_on_interrupt', True),
        'max_batch_checkpoints': kwargs.get('max_batch_checkpoints', 3),
        'resume_from_checkpoint': kwargs.get('resume_from_checkpoint', None),
        'experiment_description': kwargs.get('experiment_description', ''),
    }
    return config

def save_checkpoint(model, optimizer, epoch, best_acc, history, config, output_dir, is_best=False, 
                   current_batch=None, total_batches=None):
    """Save training checkpoint with complete state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'history': history,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'current_batch': current_batch,
        'total_batches': total_batches
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save batch checkpoint if batch info provided
    if current_batch is not None:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}_batch_{current_batch}.pt')
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        print(f"✓ Batch checkpoint saved: {checkpoint_path}")
        
        # Clean up old batch checkpoints (keep only last 3)
        cleanup_old_batch_checkpoints(output_dir, epoch, current_batch, 
                                     config.get('max_batch_checkpoints', 3))
    else:
        # Save regular epoch checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Save best model checkpoint
    if is_best:
        best_checkpoint_path = os.path.join(output_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_checkpoint_path, _use_new_zipfile_serialization=False)
        print(f"✓ Best checkpoint saved: {best_checkpoint_path}")
    
    return checkpoint_path

def cleanup_old_batch_checkpoints(output_dir, current_epoch, current_batch, max_checkpoints):
    """Clean up old batch checkpoints, keeping only the most recent ones"""
    try:
        # Find all batch checkpoints for current epoch
        batch_checkpoints = []
        for file in os.listdir(output_dir):
            if file.startswith(f'checkpoint_epoch_{current_epoch}_batch_') and file.endswith('.pt'):
                batch_num = int(file.split('_batch_')[1].split('.pt')[0])
                batch_checkpoints.append((batch_num, file))
        
        # Sort by batch number and keep only recent ones
        batch_checkpoints.sort(key=lambda x: x[0])
        if len(batch_checkpoints) > max_checkpoints:
            for batch_num, filename in batch_checkpoints[:-max_checkpoints]:
                old_path = os.path.join(output_dir, filename)
                os.remove(old_path)
                print(f"✓ Cleaned up old checkpoint: {filename}")
    except Exception as e:
        print(f"Warning: Failed to cleanup old checkpoints: {e}")

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint and return state"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Try loading with weights_only=False for backward compatibility
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load with weights_only=False, trying with weights_only=True: {e}")
        # If that fails, try with weights_only=True and add safe globals
        import numpy as np
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_acc': checkpoint['best_acc'],
        'history': checkpoint['history'],
        'config': checkpoint['config'],
        'current_batch': checkpoint.get('current_batch', None),
        'total_batches': checkpoint.get('total_batches', None)
    }

def log_experiment_change(output_dir, change_type, old_config, new_config):
    """Log when experiment configuration changes"""
    log_file = os.path.join(output_dir, 'experiment_changes.log')
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"EXPERIMENT CHANGE LOG - {datetime.now().isoformat()}\n")
        f.write(f"Change Type: {change_type}\n")
        f.write(f"{'='*60}\n")
        
        if change_type == "TRAINING_DATA_RATIO_CHANGE":
            f.write("OLD SAMPLING WEIGHTS:\n")
            if old_config.get('sampling_weights'):
                for emotion, weight in old_config['sampling_weights'].items():
                    f.write(f"  {emotion}: {weight}\n")
            else:
                f.write("  No sampling weights (uniform)\n")
            
            f.write("\nNEW SAMPLING WEIGHTS:\n")
            if new_config.get('sampling_weights'):
                for emotion, weight in new_config['sampling_weights'].items():
                    f.write(f"  {emotion}: {weight}\n")
            else:
                f.write("  No sampling weights (uniform)\n")
        
        elif change_type == "RESUME_FROM_CHECKPOINT":
            f.write(f"Resumed from: {new_config.get('resume_from_checkpoint', 'Unknown')}\n")
            f.write(f"Resume epoch: {new_config.get('resume_epoch', 'Unknown')}\n")
        
        f.write(f"\nFull new config: {json.dumps(new_config, indent=2)}\n")
        f.write(f"{'='*60}\n")

def plot_confusion_matrix(y_true, y_pred, emotion_labels, output_dir='./outputs'):
    """Plot confusion matrix"""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title('Confusion Matrix - FER2013 Benchmark')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confusion_matrix_benchmark.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved as '{output_path}'")

def run_experiment(config):
    """Run a single experiment with given configuration"""
    experiment_name = config['experiment_name']
    timestamp = config['timestamp']
    sampling_weights = config['sampling_weights']
    resume_from_checkpoint = config.get('resume_from_checkpoint')
    
    # Create experiment-specific output directory
    OUTPUT_DIR = f'./experiments/{experiment_name}_{timestamp}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize experiment log
    log_file = os.path.join(OUTPUT_DIR, 'experiment_changes.log')
    with open(log_file, 'w') as f:
        f.write(f"EXPERIMENT LOG - {experiment_name}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Description: {config.get('experiment_description', 'No description provided')}\n")
        f.write(f"{'='*60}\n")
    
    # Save experiment configuration
    config_path = os.path.join(OUTPUT_DIR, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment config saved to: {config_path}")
    
    # Check if resuming from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if resume_from_checkpoint:
        print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        # Log the resume action
        log_experiment_change(OUTPUT_DIR, "RESUME_FROM_CHECKPOINT", {}, config)
    
    MODEL_NAME = config['model_name']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']
    NUM_EPOCHS = config['num_epochs']
    SAMPLE_PER_CLASS = config['sample_per_class']
    USE_FULL_DATASET = config.get('use_full_dataset', False)
    
    # Setup signal handlers for graceful interruption
    setup_signal_handlers()
    
    # Print dataset statistics
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    
    train_stats, train_total = get_dataset_statistics(TRAIN_DIR, EMOTION_LABELS)
    print(f"\nTraining set:")
    for emotion, count in train_stats.items():
        print(f"  {emotion:10s}: {count:5d} images")
    print(f"  {'Total':10s}: {train_total:5d} images")
    
    test_stats, test_total = get_dataset_statistics(TEST_DIR, EMOTION_LABELS)
    print(f"\nTest set:")
    for emotion, count in test_stats.items():
        print(f"  {emotion:10s}: {count:5d} images")
    print(f"  {'Total':10s}: {test_total:5d} images")
    
    # Load pre-trained model and processor
    print(f"\n{'='*50}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"{'='*50}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=7,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    # Create datasets with sampling weights
    print(f"\n{'='*50}")
    if USE_FULL_DATASET:
        print("Creating datasets (FULL DATASET MODE)")
        print("⚠️  Using conservative batch size for memory safety")
    else:
        print(f"Creating datasets ({SAMPLE_PER_CLASS} samples per class)")
    if sampling_weights:
        print(f"Sampling weights: {sampling_weights}")
    print(f"{'='*50}")
    
    train_dataset = FER2013FolderDataset(
        TRAIN_DIR, 
        processor, 
        EMOTION_LABELS,
        max_samples_per_class=None if USE_FULL_DATASET else SAMPLE_PER_CLASS,
        sampling_weights=sampling_weights,
        use_full_dataset=USE_FULL_DATASET
    )
    
    # Use smaller test set for faster benchmarking
    test_dataset = FER2013FolderDataset(
        TEST_DIR, 
        processor, 
        EMOTION_LABELS,
        max_samples_per_class=100,
        use_full_dataset=False  # Keep test set small for now
    )
    
    # Create dataloaders with weighted sampling if specified
    # For full dataset mode, avoid WeightedRandomSampler to prevent bias
    if sampling_weights and not USE_FULL_DATASET:
        sampler_weights = train_dataset.get_sampling_weights()
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if resuming
    if resume_from_checkpoint:
        try:
            checkpoint_state = load_checkpoint(resume_from_checkpoint, model, optimizer)
            start_epoch = checkpoint_state['epoch'] + 1  # Resume from next epoch
            best_val_acc = checkpoint_state['best_acc']
            history = checkpoint_state['history']
            print(f"✓ Resumed from epoch {checkpoint_state['epoch']} with best acc: {best_val_acc:.4f}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("Starting fresh training...")
            start_epoch = 0
            best_val_acc = 0.0
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Training loop
    print(f"\n{'='*50}")
    print("Starting benchmark training...")
    if resume_from_checkpoint:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'='*50}\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train with enhanced checkpointing
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config=config, output_dir=OUTPUT_DIR, epoch=epoch,
            best_acc=best_val_acc, history=history
        )
        
        # Check if training was interrupted
        if TRAINING_INTERRUPTED:
            print("Training was interrupted. Exiting...")
            break
        
        # Evaluate
        val_preds, val_labels, val_loss = evaluate(model, test_loader, device)
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Print memory usage
        memory_info = get_memory_usage()
        print(f"GPU Memory: {memory_info['allocated']:.1f}GB allocated, {memory_info['reserved']:.1f}GB reserved")
        
        # Save checkpoint with complete state
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"✓ New best model! (Val Acc: {val_acc:.4f})")
        
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch+1, best_val_acc, history, config, OUTPUT_DIR, is_best
        )
        
        # Also save individual epoch model for compatibility
        epoch_model_path = os.path.join(OUTPUT_DIR, f'epoch_{epoch+1}_model.pt')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"✓ Epoch {epoch+1} model saved to: {epoch_model_path}")
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*50}")
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    best_checkpoint_path = os.path.join(OUTPUT_DIR, 'best_checkpoint.pt')
    if os.path.exists(best_checkpoint_path):
        print("Loading best checkpoint for final evaluation...")
        load_checkpoint(best_checkpoint_path, model)
    else:
        print("Loading best model state for final evaluation...")
        # Fallback to old format if best_checkpoint doesn't exist
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("No best model found, using current model state")
    
    final_preds, final_labels, _ = evaluate(model, test_loader, device)
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(final_labels, final_preds, 
                                   target_names=EMOTION_LABELS, 
                                   digits=4)
    print(report)
    
    # Save classification report
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nClassification report saved to '{report_path}'")
    
    # Plot confusion matrix
    plot_confusion_matrix(final_labels, final_preds, EMOTION_LABELS, OUTPUT_DIR)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    history_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(history_path)
    plt.close()
    print(f"Training history saved as '{history_path}'")
    
    # Save training history data
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history data saved to '{history_csv_path}'")
    
    print(f"\n{'='*50}")
    print("Experiment complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*50}")
    
    return OUTPUT_DIR, best_val_acc

def main():
    """Main function to run experiments"""
    print("Emotion Recognition Experiment Runner")
    print("=" * 50)
    
    # Define experiment configurations
    experiments = [
        # FULL DATASET EXPERIMENT - Multi-epoch for training curves
        create_experiment_config(
            experiment_name="full_dataset_multi_epoch",
            sampling_weights=None,
            use_full_dataset=True,
            batch_size=16,  # Conservative batch size for memory safety
            num_epochs=5,   # Multiple epochs for proper training curves
            batch_checkpoint_frequency=100,  # Checkpoint every 100 batches
            experiment_description="Full dataset training with 28,709 images - 3 epochs with enhanced checkpointing and training curves"
        )
    ]
    
    # Run experiments
    results = []
    for i, config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Running Experiment {i+1}/{len(experiments)}: {config['experiment_name']}")
        print(f"{'='*60}")
        
        try:
            output_dir, best_acc = run_experiment(config)
            results.append({
                'experiment': config['experiment_name'],
                'output_dir': output_dir,
                'best_accuracy': best_acc,
                'sampling_weights': config['sampling_weights']
            })
            print(f"✓ Experiment {config['experiment_name']} completed successfully!")
        except Exception as e:
            print(f"✗ Experiment {config['experiment_name']} failed: {e}")
            results.append({
                'experiment': config['experiment_name'],
                'output_dir': None,
                'best_accuracy': 0.0,
                'sampling_weights': config['sampling_weights'],
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for result in results:
        if 'error' in result:
            print(f"❌ {result['experiment']}: FAILED - {result['error']}")
        else:
            print(f"✅ {result['experiment']}: {result['best_accuracy']:.4f} accuracy")
            print(f"   Output: {result['output_dir']}")
    
    print(f"\nAll experiments completed!")

if __name__ == "__main__":
    main()