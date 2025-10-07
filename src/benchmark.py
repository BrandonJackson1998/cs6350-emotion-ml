# benchmark.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths - relative to project root
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Emotion labels (alphabetically ordered to match folder structure)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}

class FER2013FolderDataset(Dataset):
    """Custom Dataset for FER2013 organized in folders"""
    
    def __init__(self, root_dir, processor, emotion_labels=EMOTION_LABELS, max_samples_per_class=None):
        self.root_dir = root_dir
        self.processor = processor
        self.emotion_labels = emotion_labels
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for emotion in emotion_labels:
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit samples per class if specified (for benchmarking)
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for img_file in image_files:
                self.image_paths.append(os.path.join(emotion_dir, img_file))
                self.labels.append(self.emotion_to_idx[emotion])
        
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
        
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

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
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
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
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

def main():
    # Configuration
    MODEL_NAME = "dima806/facial_emotions_image_detection"
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    SAMPLE_PER_CLASS = 100  # For benchmark, use 100 samples per class
    OUTPUT_DIR = './outputs'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    
    # Create datasets (benchmark with limited samples)
    print(f"\n{'='*50}")
    print(f"Creating benchmark datasets ({SAMPLE_PER_CLASS} samples per class)")
    print(f"{'='*50}")
    train_dataset = FER2013FolderDataset(
        TRAIN_DIR, 
        processor, 
        EMOTION_LABELS,
        max_samples_per_class=SAMPLE_PER_CLASS
    )
    
    # Use smaller test set for faster benchmarking
    test_dataset = FER2013FolderDataset(
        TEST_DIR, 
        processor, 
        EMOTION_LABELS,
        max_samples_per_class=100
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n{'='*50}")
    print("Starting benchmark training...")
    print(f"{'='*50}\n")
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(OUTPUT_DIR, 'best_model_benchmark.pt')
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model_benchmark.pt')))
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
    history_path = os.path.join(OUTPUT_DIR, 'training_history_benchmark.png')
    plt.savefig(history_path)
    plt.close()
    print(f"Training history saved as '{history_path}'")
    
    # Save training history data
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history data saved to '{history_csv_path}'")
    
    print(f"\n{'='*50}")
    print("Benchmark complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()