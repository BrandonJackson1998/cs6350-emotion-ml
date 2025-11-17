#!/usr/bin/env python3
"""
Adaptive Emotion-Focused Training
Iteratively identifies the two worst-performing emotions and trains on them for 10 epochs.
Each epoch: identify worst 2 emotions → train on those → test on all → repeat
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark import (
    FER2013FolderDataset,
    train_epoch,
    evaluate,
    plot_confusion_matrix,
    save_checkpoint,
    load_checkpoint,
    EMOTION_LABELS,
    TRAIN_DIR,
    TEST_DIR,
    device
)

from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader
from PIL import Image

class FocusedDataset(torch.utils.data.Dataset):
    """Dataset that focuses on specific emotions only"""
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            image = Image.new('RGB', (48, 48), color='black')
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def parse_classification_report(report_text):
    """Parse classification report to extract per-class accuracy (f1-scores)"""
    lines = report_text.strip().split('\n')
    emotion_scores = {}
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            emotion = parts[0]
            if emotion in EMOTION_LABELS:
                try:
                    f1_score = float(parts[3])
                    emotion_scores[emotion] = f1_score
                except (ValueError, IndexError):
                    continue
    
    return emotion_scores

def get_worst_two_emotions(emotion_scores):
    """Get the two emotions with lowest f1-scores"""
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1])
    worst_two = [emotion for emotion, score in sorted_emotions[:2]]
    return worst_two

def generate_adaptive_training_plots(history, output_dir):
    """Generate comprehensive visualization of adaptive training progress"""
    
    epochs = history['epoch']
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Overall Test Accuracy Over Epochs
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['test_acc'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Overall Test Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(history['test_acc']) - 0.05, max(history['test_acc']) + 0.05])
    
    # Add value labels
    for i, (e, acc) in enumerate(zip(epochs, history['test_acc'])):
        ax1.text(e, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
    
    # Plot 2: Training Accuracy Over Epochs
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy (Focused Emotions)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss Over Epochs
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['train_loss'], 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Per-Emotion F1-Score Evolution
    ax4 = plt.subplot(2, 3, 4)
    
    # Extract emotion scores for each epoch
    emotion_progression = {emotion: [] for emotion in EMOTION_LABELS}
    for epoch_scores in history['emotion_scores']:
        for emotion in EMOTION_LABELS:
            emotion_progression[emotion].append(epoch_scores.get(emotion, 0))
    
    # Plot each emotion
    colors = plt.cm.tab10(np.linspace(0, 1, len(EMOTION_LABELS)))
    for i, (emotion, scores) in enumerate(emotion_progression.items()):
        ax4.plot(epochs, scores, 'o-', label=emotion, linewidth=2, 
                markersize=6, alpha=0.8, color=colors[i])
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('F1-Score', fontsize=12)
    ax4.set_title('Per-Emotion F1-Score Evolution', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Plot 5: Focus Emotions Timeline
    ax5 = plt.subplot(2, 3, 5)
    
    # Create a text summary of which emotions were focused on each epoch
    focus_text = "Epochs and Focus Emotions:\n" + "-" * 40 + "\n"
    for i, (epoch, focus) in enumerate(zip(epochs, history['focus_emotions'])):
        focus_text += f"Epoch {epoch}: {', '.join(focus)}\n"
    
    ax5.text(0.1, 0.5, focus_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax5.axis('off')
    ax5.set_title('Adaptive Focus Timeline', fontsize=14, fontweight='bold')
    
    # Plot 6: Improvement Heatmap
    ax6 = plt.subplot(2, 3, 6)
    
    # Create matrix of emotion scores across epochs
    emotion_matrix = []
    for emotion in EMOTION_LABELS:
        scores = [epoch_scores.get(emotion, 0) for epoch_scores in history['emotion_scores']]
        emotion_matrix.append(scores)
    
    emotion_matrix = np.array(emotion_matrix)
    
    im = ax6.imshow(emotion_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Emotion', fontsize=12)
    ax6.set_title('F1-Score Heatmap', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(epochs)))
    ax6.set_xticklabels(epochs)
    ax6.set_yticks(range(len(EMOTION_LABELS)))
    ax6.set_yticklabels(EMOTION_LABELS)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('F1-Score', fontsize=10)
    
    # Add text annotations
    for i in range(len(EMOTION_LABELS)):
        for j in range(len(epochs)):
            text = ax6.text(j, i, f'{emotion_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.suptitle('Adaptive Emotion-Focused Training Progress', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(output_dir, 'adaptive_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Progress visualization saved: adaptive_training_progress.png")
    
    # Generate additional individual plots for clarity
    
    # Individual plot: Overall accuracy trend
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['test_acc'], 'o-', linewidth=3, markersize=10, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.title('Overall Test Accuracy Over Adaptive Training', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (e, acc) in enumerate(zip(epochs, history['test_acc'])):
        plt.text(e, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    accuracy_plot_path = os.path.join(output_dir, 'test_accuracy_progression.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Accuracy plot saved: test_accuracy_progression.png")

def adaptive_training(initial_model_path, num_epochs=10, batch_size=16):
    """
    Adaptive training: each epoch trains on the 2 worst-performing emotions
    """
    print("🎯 Adaptive Emotion-Focused Training")
    print("=" * 80)
    print(f"Initial Model: {initial_model_path}")
    print(f"Training Epochs: {num_epochs}")
    print(f"Strategy: Train on worst 2 emotions each epoch, evaluate on all")
    print("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./experiments/adaptive_training_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor and model
    MODEL_NAME = "dima806/facial_emotions_image_detection"
    print(f"\n📥 Loading model and processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=7,
        ignore_mismatched_sizes=True
    )
    
    # Load the trained model weights from initial run
    print(f"Loading initial trained weights from: {initial_model_path}")
    state_dict = torch.load(initial_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("✓ Model loaded")
    
    # Create full test dataset (always evaluate on all test data)
    print(f"\n📂 Loading test dataset (full)...")
    test_dataset = FER2013FolderDataset(
        TEST_DIR,
        processor,
        EMOTION_LABELS,
        use_full_dataset=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} images")
    
    # Initial evaluation to get baseline worst emotions
    print(f"\n📊 Initial evaluation on full test set...")
    initial_preds, initial_labels, _ = evaluate(model, test_loader, device)
    initial_report = classification_report(
        initial_labels, 
        initial_preds,
        target_names=EMOTION_LABELS,
        digits=4
    )
    print(initial_report)
    
    # Parse to get emotion scores
    emotion_scores = parse_classification_report(initial_report)
    
    # Training loop
    history = {
        'epoch': [],
        'focus_emotions': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'emotion_scores': []
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    best_overall_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Identify worst 2 emotions
        worst_emotions = get_worst_two_emotions(emotion_scores)
        print(f"\n🎯 Focus Emotions for this epoch: {worst_emotions}")
        print(f"   Current F1-scores:")
        for emotion in worst_emotions:
            print(f"   - {emotion}: {emotion_scores[emotion]:.4f}")
        
        # Create training dataset with 70% focus emotions, 30% other emotions
        print(f"\n📂 Creating mixed training dataset (70% focus, 30% maintenance)...")
        
        # Collect images for focus emotions (70% of dataset)
        focus_images = []
        focus_labels = []
        
        for emotion in worst_emotions:
            emotion_dir = os.path.join(TRAIN_DIR, emotion)
            if os.path.exists(emotion_dir):
                image_files = [f for f in os.listdir(emotion_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                emotion_idx = EMOTION_LABELS.index(emotion)
                
                for img_file in image_files:
                    focus_images.append(os.path.join(emotion_dir, img_file))
                    focus_labels.append(emotion_idx)
        
        # Collect images for maintenance emotions (30% of dataset)
        # Calculate how many maintenance samples we need based on focus samples
        num_focus_samples = len(focus_images)
        num_maintenance_samples = int(num_focus_samples * 0.3 / 0.7)  # 30% of total dataset
        
        maintenance_emotions = [e for e in EMOTION_LABELS if e not in worst_emotions]
        samples_per_maintenance_emotion = num_maintenance_samples // len(maintenance_emotions)
        
        maintenance_images = []
        maintenance_labels = []
        
        import random
        for emotion in maintenance_emotions:
            emotion_dir = os.path.join(TRAIN_DIR, emotion)
            if os.path.exists(emotion_dir):
                image_files = [f for f in os.listdir(emotion_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                emotion_idx = EMOTION_LABELS.index(emotion)
                
                # Sample a subset for maintenance
                sampled_files = random.sample(image_files, 
                                             min(samples_per_maintenance_emotion, len(image_files)))
                
                for img_file in sampled_files:
                    maintenance_images.append(os.path.join(emotion_dir, img_file))
                    maintenance_labels.append(emotion_idx)
        
        # Combine focus and maintenance datasets
        train_images = focus_images + maintenance_images
        train_labels = focus_labels + maintenance_labels
        
        print(f"✓ Mixed training set created:")
        print(f"   Focus emotions ({', '.join(worst_emotions)}): {len(focus_images)} images ({len(focus_images)/len(train_images)*100:.1f}%)")
        print(f"   Maintenance emotions ({', '.join(maintenance_emotions)}): {len(maintenance_images)} images ({len(maintenance_images)/len(train_images)*100:.1f}%)")
        print(f"   Total: {len(train_images)} images")
        
        # Create focused dataset
        focused_dataset = FocusedDataset(train_images, train_labels, processor)
        focused_loader = DataLoader(focused_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Train on mixed dataset
        print(f"\n🏋️ Training on focus emotions with maintenance...")
        train_loss, train_acc = train_epoch(
            model, focused_loader, optimizer, criterion, device
        )
        
        print(f"\nTraining Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f}")
        
        # Evaluate on FULL test set
        print(f"\n📊 Evaluating on full test set...")
        test_preds, test_labels, test_loss = evaluate(model, test_loader, device)
        test_acc = (np.array(test_preds) == np.array(test_labels)).mean()
        
        # Get detailed per-emotion scores
        report = classification_report(
            test_labels,
            test_preds,
            target_names=EMOTION_LABELS,
            digits=4
        )
        print(f"\nTest Results:")
        print(f"  Overall Accuracy: {test_acc:.4f}")
        print(f"\nDetailed Classification Report:")
        print(report)
        
        # Update emotion scores for next iteration
        emotion_scores = parse_classification_report(report)
        
        # Track history
        history['epoch'].append(epoch + 1)
        history['focus_emotions'].append(worst_emotions)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['emotion_scores'].append(emotion_scores.copy())
        
        # Save checkpoint
        is_best = test_acc > best_overall_acc
        if is_best:
            best_overall_acc = test_acc
            print(f"✓ New best overall accuracy: {best_overall_acc:.4f}")
        
        config = {
            'experiment_name': 'adaptive_training',
            'epoch': epoch + 1,
            'focus_emotions': worst_emotions,
            'batch_size': batch_size,
            'learning_rate': 2e-5
        }
        
        save_checkpoint(
            model, optimizer, epoch, best_overall_acc, history, config,
            output_dir, is_best=is_best
        )
        
        # Save epoch model
        epoch_model_path = os.path.join(output_dir, f'epoch_{epoch+1}_model.pt')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"✓ Saved epoch {epoch+1} model")
        
        # Save classification report
        report_path = os.path.join(output_dir, f'classification_report_epoch_{epoch+1}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Epoch {epoch+1}\n")
            f.write(f"Focus Emotions: {worst_emotions}\n")
            f.write(f"Overall Test Accuracy: {test_acc:.4f}\n\n")
            f.write(report)
        
        # Save confusion matrix
        plot_confusion_matrix(test_labels, test_preds, EMOTION_LABELS, output_dir)
        import shutil
        shutil.move(
            os.path.join(output_dir, 'confusion_matrix_benchmark.png'),
            os.path.join(output_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        )
    
    # Final summary
    print(f"\n{'='*80}")
    print("ADAPTIVE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest Overall Accuracy: {best_overall_acc:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'adaptive_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate visualization graphs
    print(f"\n📊 Generating progress visualization...")
    generate_adaptive_training_plots(history, output_dir)
    
    # Generate summary report
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Adaptive Emotion-Focused Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Best Overall Accuracy: {best_overall_acc:.4f}\n\n")
        f.write("Epoch-by-Epoch Progress:\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(history['epoch'])):
            f.write(f"\nEpoch {history['epoch'][i]}:\n")
            f.write(f"  Focus Emotions: {history['focus_emotions'][i]}\n")
            f.write(f"  Train Loss: {history['train_loss'][i]:.4f}\n")
            f.write(f"  Train Acc: {history['train_acc'][i]:.4f}\n")
            f.write(f"  Test Acc: {history['test_acc'][i]:.4f}\n")
            f.write(f"  Emotion Scores:\n")
            for emotion, score in history['emotion_scores'][i].items():
                f.write(f"    {emotion}: {score:.4f}\n")
    
    print(f"✓ Training summary saved to: {summary_path}")
    
    return output_dir, best_overall_acc

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Adaptive Emotion-Focused Training"
    )
    parser.add_argument(
        '--initial-model',
        default='experiments/full_dataset_single_epoch_20251114_090055/epoch_1_model.pt',
        help='Path to initial trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of adaptive training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.initial_model):
        print(f"❌ Initial model not found: {args.initial_model}")
        sys.exit(1)
    
    try:
        output_dir, best_acc = adaptive_training(
            args.initial_model,
            args.epochs,
            args.batch_size
        )
        print(f"\n✅ Adaptive training completed successfully!")
        print(f"Best accuracy achieved: {best_acc:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
