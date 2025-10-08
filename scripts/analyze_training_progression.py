#!/usr/bin/env python3
"""
Training Progression Analyzer

This script analyzes training progression across multiple experiments,
showing how the model evolved with different emotion focus lenses.
It can coalesce data from checkpoint chains and visualize the training journey.
"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

def load_experiment_data(experiment_dir):
    """Load experiment data from a directory."""
    data = {}
    
    # Load experiment config
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    
    # Load training history
    history_path = os.path.join(experiment_dir, 'training_history.csv')
    if os.path.exists(history_path):
        data['history'] = pd.read_csv(history_path)
    
    # Load classification report
    report_path = os.path.join(experiment_dir, 'classification_report.txt')
    if os.path.exists(report_path):
        data['report_path'] = report_path
    
    return data

def find_experiment_chain(start_experiment_dir):
    """Find the chain of experiments starting from a checkpoint."""
    chain = []
    current_dir = start_experiment_dir
    
    while current_dir and os.path.exists(current_dir):
        data = load_experiment_data(current_dir)
        if data:
            chain.append({
                'directory': current_dir,
                'data': data
            })
        
        # Look for next experiment in the chain
        # This would need to be implemented based on your naming convention
        # For now, we'll just return the current experiment
        break
    
    return chain

def create_training_progression_plot(experiments_data, output_path):
    """Create a comprehensive training progression visualization."""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progression Analysis', fontsize=16, fontweight='bold')
    
    # Colors for different experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Training Loss Progression
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments_data):
        if 'history' in exp['data']:
            history = exp['data']['history']
            exp_name = os.path.basename(exp['directory']).split('_')[0]  # Get experiment name
            epochs = range(1, len(history) + 1)  # Create epoch numbers
            ax1.plot(epochs, history['train_loss'], 
                    label=f"{exp_name} (Loss)", color=colors[i % len(colors)], linewidth=2)
            ax1.plot(epochs, history['val_loss'], 
                    label=f"{exp_name} (Val Loss)", color=colors[i % len(colors)], 
                    linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy Progression
    ax2 = axes[0, 1]
    for i, exp in enumerate(experiments_data):
        if 'history' in exp['data']:
            history = exp['data']['history']
            exp_name = os.path.basename(exp['directory']).split('_')[0]
            epochs = range(1, len(history) + 1)  # Create epoch numbers
            ax2.plot(epochs, history['train_acc'], 
                    label=f"{exp_name} (Train Acc)", color=colors[i % len(colors)], linewidth=2)
            ax2.plot(epochs, history['val_acc'], 
                    label=f"{exp_name} (Val Acc)", color=colors[i % len(colors)], 
                    linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy Progression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Emotion Focus Visualization
    ax3 = axes[1, 0]
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Create a heatmap of emotion weights across experiments
    weight_matrix = []
    exp_names = []
    
    for exp in experiments_data:
        if 'config' in exp['data']:
            config = exp['data']['config']
            weights = config.get('sampling_weights', {})
            weight_row = [weights.get(emotion, 1.0) for emotion in emotion_names]
            weight_matrix.append(weight_row)
            exp_names.append(os.path.basename(exp['directory']).split('_')[0])
    
    if weight_matrix:
        weight_df = pd.DataFrame(weight_matrix, index=exp_names, columns=emotion_names)
        sns.heatmap(weight_df, annot=True, cmap='RdYlBu_r', center=1.0, 
                   ax=ax3, cbar_kws={'label': 'Sampling Weight'})
        ax3.set_title('Emotion Focus Across Experiments')
        ax3.set_xlabel('Emotions')
        ax3.set_ylabel('Experiments')
    
    # Plot 4: Performance Summary
    ax4 = axes[1, 1]
    
    # Get final accuracies and create a bar chart
    final_accs = []
    exp_labels = []
    
    for exp in experiments_data:
        if 'history' in exp['data']:
            history = exp['data']['history']
            final_acc = history['val_acc'].iloc[-1] if len(history) > 0 else 0
            final_accs.append(final_acc)
            exp_labels.append(os.path.basename(exp['directory']).split('_')[0])
    
    if final_accs:
        bars = ax4.bar(exp_labels, final_accs, color=colors[:len(final_accs)])
        ax4.set_ylabel('Final Validation Accuracy')
        ax4.set_title('Final Performance Comparison')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_emotion_focus_timeline(experiments_data, output_path):
    """Create a timeline showing emotion focus changes."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for timeline
    timeline_data = []
    
    for i, exp in enumerate(experiments_data):
        if 'config' in exp['data']:
            config = exp['data']['config']
            weights = config.get('sampling_weights', {})
            
            # Extract timestamp from directory name
            dir_name = os.path.basename(exp['directory'])
            timestamp_part = '_'.join(dir_name.split('_')[-2:])  # Get timestamp part
            
            timeline_data.append({
                'experiment': dir_name.split('_')[0],
                'timestamp': timestamp_part,
                'weights': weights,
                'epoch_start': i * 10,  # Approximate epoch start
                'epoch_end': (i + 1) * 10  # Approximate epoch end
            })
    
    # Create emotion focus bars
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71', '#95a5a6', '#3498db', '#e67e22']
    
    y_pos = 0
    for i, exp_data in enumerate(timeline_data):
        weights = exp_data['weights']
        
        # Create horizontal bars for each emotion
        for j, emotion in enumerate(emotions):
            weight = weights.get(emotion, 1.0)
            width = weight * 2  # Scale for visibility
            
            # Color intensity based on weight
            alpha = min(weight / 3.0, 1.0)  # Cap at 1.0
            
            ax.barh(y_pos + j * 0.1, width, height=0.08, 
                   color=colors[j], alpha=alpha, 
                   label=emotion if i == 0 else "")
        
        # Add experiment label
        ax.text(-0.5, y_pos + 3.5, exp_data['experiment'], 
               rotation=0, va='center', fontweight='bold')
        
        y_pos += 1
    
    ax.set_xlabel('Sampling Weight (scaled)')
    ax.set_ylabel('Experiments')
    ax.set_title('Emotion Focus Timeline Across Experiments')
    ax.set_xlim(-1, 8)
    
    # Create custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=colors[i], label=emotion) 
                      for i, emotion in enumerate(emotions)]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_analysis_report(experiments_data, output_path):
    """Generate a comprehensive analysis report."""
    
    report = []
    report.append("# Training Progression Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    report.append("## Experiment Summary")
    report.append("")
    
    for i, exp in enumerate(experiments_data):
        dir_name = os.path.basename(exp['directory'])
        report.append(f"### Experiment {i+1}: {dir_name}")
        
        if 'config' in exp['data']:
            config = exp['data']['config']
            report.append(f"- **Description**: {config.get('experiment_description', 'N/A')}")
            report.append(f"- **Epochs**: {config.get('num_epochs', 'N/A')}")
            report.append(f"- **Resume from**: {config.get('resume_from_checkpoint', 'New experiment')}")
            
            # Emotion weights
            weights = config.get('sampling_weights', {})
            report.append("- **Emotion Focus**:")
            for emotion, weight in weights.items():
                if weight != 1.0:  # Only show non-default weights
                    focus_type = "Promoted" if weight > 1.0 else "Demoted"
                    report.append(f"  - {emotion}: {weight}x ({focus_type})")
        
        if 'history' in exp['data']:
            history = exp['data']['history']
            if len(history) > 0:
                final_acc = history['val_acc'].iloc[-1]
                best_acc = history['val_acc'].max()
                report.append(f"- **Final Accuracy**: {final_acc:.4f}")
                report.append(f"- **Best Accuracy**: {best_acc:.4f}")
        
        report.append("")
    
    # Training progression analysis
    report.append("## Training Progression Analysis")
    report.append("")
    
    if len(experiments_data) > 1:
        report.append("### Key Observations:")
        report.append("")
        
        # Compare final accuracies
        accuracies = []
        for exp in experiments_data:
            if 'history' in exp['data']:
                history = exp['data']['history']
                if len(history) > 0:
                    accuracies.append(history['val_acc'].iloc[-1])
        
        if len(accuracies) > 1:
            improvement = accuracies[-1] - accuracies[0]
            report.append(f"- **Overall Improvement**: {improvement:+.4f} ({improvement/accuracies[0]*100:+.1f}%)")
            
            if improvement > 0:
                report.append("- **Training Success**: Model performance improved across experiments")
            else:
                report.append("- **Training Challenge**: Model performance decreased, may need different approach")
        
        report.append("")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Analyze training progression across experiments')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Paths to experiment directories to analyze')
    parser.add_argument('--output-dir', default='./analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--name', default='training_progression',
                       help='Base name for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment data
    experiments_data = []
    for exp_dir in args.experiments:
        if os.path.exists(exp_dir):
            data = load_experiment_data(exp_dir)
            if data:
                experiments_data.append({
                    'directory': exp_dir,
                    'data': data
                })
                print(f"✅ Loaded experiment: {exp_dir}")
            else:
                print(f"⚠️  No data found in: {exp_dir}")
        else:
            print(f"❌ Directory not found: {exp_dir}")
    
    if not experiments_data:
        print("❌ No valid experiments found!")
        return
    
    print(f"\n📊 Analyzing {len(experiments_data)} experiments...")
    
    # Generate visualizations
    progression_plot = create_training_progression_plot(
        experiments_data, 
        os.path.join(args.output_dir, f"{args.name}_progression.png")
    )
    
    timeline_plot = create_emotion_focus_timeline(
        experiments_data,
        os.path.join(args.output_dir, f"{args.name}_timeline.png")
    )
    
    # Generate report
    report_path = generate_analysis_report(
        experiments_data,
        os.path.join(args.output_dir, f"{args.name}_report.md")
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Progression plot: {progression_plot}")
    print(f"📈 Timeline plot: {timeline_plot}")
    print(f"📝 Report: {report_path}")

if __name__ == "__main__":
    main()
