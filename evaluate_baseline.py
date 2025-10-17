#!/usr/bin/env python3
"""
Baseline Model Evaluation Script
Measures the performance of the pre-trained model on the test dataset without any fine-tuning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Import from our benchmark module
from src.benchmark import (
    FER2013FolderDataset, 
    evaluate, 
    plot_confusion_matrix,
    TEST_DIR,
    device
)

# Use the correct label order that the model expects
# Model expects: sad, disgust, angry, neutral, fear, surprise, happy
EMOTION_LABELS = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

def evaluate_baseline_model():
    """Evaluate the pre-trained model without any fine-tuning"""
    
    print("🔍 Baseline Model Evaluation")
    print("=" * 60)
    print(f"Model: dima806/facial_emotions_image_detection")
    print(f"Dataset: FER2013 Test Set")
    print(f"Evaluation: Pre-trained model (no fine-tuning)")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load pre-trained model and processor (no fine-tuning)
    MODEL_NAME = "dima806/facial_emotions_image_detection"
    print(f"Loading pre-trained model: {MODEL_NAME}")
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=7,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Model loaded successfully")
    
    # Create test dataset (use full test set)
    print("Loading test dataset...")
    test_dataset = FER2013FolderDataset(
        TEST_DIR,
        processor,
        EMOTION_LABELS,
        use_full_dataset=True  # Use all test data
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} images")
    print("\nClass distribution in test set:")
    for emotion, count in test_dataset.class_counts.items():
        print(f"  {emotion:10s}: {count:5d} images")
    
    # Evaluate baseline model
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE MODEL (NO TRAINING)")
    print("=" * 60)
    
    # Run evaluation
    predictions, true_labels, avg_loss = evaluate(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n📊 BASELINE MODEL RESULTS")
    print("-" * 40)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Total Test Images: {len(predictions)}")
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    report = classification_report(
        true_labels, 
        predictions,
        target_names=EMOTION_LABELS,
        digits=4
    )
    print(report)
    
    # Create output directory
    output_dir = f"./outputs/baseline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results = {
        'model_name': MODEL_NAME,
        'evaluation_type': 'baseline_pretrained',
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': float(accuracy),
        'average_loss': float(avg_loss),
        'total_images': len(predictions),
        'class_distribution': test_dataset.class_counts,
        'detailed_metrics': {}
    }
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=list(range(len(EMOTION_LABELS)))
    )
    
    print(f"\n📈 PER-CLASS PERFORMANCE ANALYSIS")
    print("-" * 70)
    print(f"{'Emotion':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    for i, emotion in enumerate(EMOTION_LABELS):
        print(f"{emotion:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        results['detailed_metrics'][emotion] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'baseline_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save classification report
    report_path = os.path.join(output_dir, 'baseline_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Baseline Model Evaluation Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset: FER2013 Test Set ({len(predictions)} images)\n\n")
        f.write(report)
    
    # Create and save confusion matrix
    print(f"\n🎯 GENERATING CONFUSION MATRIX")
    print("-" * 40)
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.title('Baseline Model Confusion Matrix\n(Pre-trained, No Fine-tuning)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    confusion_path = os.path.join(output_dir, 'baseline_confusion_matrix.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance comparison preparation
    print(f"\n📊 PERFORMANCE INSIGHTS")
    print("-" * 50)
    
    # Find best and worst performing classes
    f1_scores = [(emotion, f1[i]) for i, emotion in enumerate(EMOTION_LABELS)]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Best performing emotions:")
    for i, (emotion, score) in enumerate(f1_scores[:3]):
        print(f"  {i+1}. {emotion}: {score:.4f} F1-score")
    
    print(f"\nWorst performing emotions:")
    for i, (emotion, score) in enumerate(f1_scores[-3:]):
        print(f"  {len(f1_scores)-i}. {emotion}: {score:.4f} F1-score")
    
    # Calculate class imbalance impact
    class_ratios = {}
    total_samples = sum(support)
    for i, emotion in enumerate(EMOTION_LABELS):
        class_ratios[emotion] = support[i] / total_samples
    
    print(f"\nClass distribution impact:")
    for emotion in EMOTION_LABELS:
        class_pct = class_ratios[emotion] * 100
        class_f1 = f1[EMOTION_LABELS.index(emotion)]
        print(f"  {emotion}: {class_pct:.1f}% of data, {class_f1:.4f} F1-score")
    
    print(f"\n✅ EVALUATION COMPLETE")
    print("-" * 40)
    print(f"Results saved to: {output_dir}")
    print(f"- Detailed results: baseline_evaluation_results.json")
    print(f"- Classification report: baseline_classification_report.txt")
    print(f"- Confusion matrix: baseline_confusion_matrix.png")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"- Baseline accuracy: {accuracy*100:.2f}%")
    print(f"- Best class: {f1_scores[0][0]} ({f1_scores[0][1]:.4f} F1)")
    print(f"- Worst class: {f1_scores[-1][0]} ({f1_scores[-1][1]:.4f} F1)")
    print(f"- Model shows {'strong' if accuracy > 0.6 else 'moderate' if accuracy > 0.4 else 'weak'} baseline performance")
    
    return results, output_dir

if __name__ == "__main__":
    try:
        results, output_dir = evaluate_baseline_model()
        print(f"\n🚀 Baseline evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)