#!/usr/bin/env python3
"""
Training Lens Visualizer

This script shows the training lens (emotion focus) progression across experiments,
making it easy to see how the model's training focus evolved over time.
"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_experiment_config(experiment_dir):
    """Load experiment configuration."""
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def create_lens_progression_plot(experiments, output_path):
    """Create a visualization showing how the training lens evolved."""
    
    # Set up the plot
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Colors for emotions
    emotion_colors = {
        'angry': '#e74c3c',
        'disgust': '#f39c12', 
        'fear': '#9b59b6',
        'happy': '#2ecc71',
        'neutral': '#95a5a6',
        'sad': '#3498db',
        'surprise': '#e67e22'
    }
    
    emotions = list(emotion_colors.keys())
    
    # Prepare data
    experiment_names = []
    weight_data = []
    
    for exp_dir in experiments:
        config = load_experiment_config(exp_dir)
        if config:
            exp_name = os.path.basename(exp_dir).split('_')[0]
            experiment_names.append(exp_name)
            
            weights = config.get('sampling_weights', {})
            weight_row = [weights.get(emotion, 1.0) for emotion in emotions]
            weight_data.append(weight_row)
    
    if not weight_data:
        print("❌ No experiment data found!")
        return
    
    # Plot 1: Heatmap of emotion weights
    weight_df = pd.DataFrame(weight_data, index=experiment_names, columns=emotions)
    
    # Create custom colormap centered at 1.0
    sns.heatmap(weight_df, annot=True, cmap='RdYlBu_r', center=1.0, 
                ax=ax1, cbar_kws={'label': 'Sampling Weight'}, 
                fmt='.1f', square=True)
    
    ax1.set_title('Training Lens Evolution: Emotion Focus Across Experiments', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Emotions')
    ax1.set_ylabel('Experiments')
    
    # Add annotations for focus types
    for i, row in enumerate(weight_data):
        for j, weight in enumerate(row):
            if weight > 1.5:
                ax1.text(j + 0.5, i + 0.7, '↑', ha='center', va='center', 
                        fontsize=12, color='darkred', fontweight='bold')
            elif weight < 0.7:
                ax1.text(j + 0.5, i + 0.3, '↓', ha='center', va='center', 
                        fontsize=12, color='darkblue', fontweight='bold')
    
    # Plot 2: Timeline of emotion focus changes
    ax2.set_xlim(0, len(experiments))
    ax2.set_ylim(0, 7)
    
    # Create timeline bars
    for i, (exp_name, weights) in enumerate(zip(experiment_names, weight_data)):
        y_pos = 0
        for j, (emotion, weight) in enumerate(zip(emotions, weights)):
            # Bar width proportional to weight
            width = weight * 0.3
            height = 0.8
            
            # Color intensity based on weight
            alpha = min(weight / 3.0, 1.0)
            color = emotion_colors[emotion]
            
            # Draw bar
            rect = plt.Rectangle((i, y_pos), width, height, 
                               facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax2.add_patch(rect)
            
            # Add weight label
            if weight != 1.0:
                ax2.text(i + width/2, y_pos + height/2, f'{weight:.1f}x', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            y_pos += 1
    
    # Customize timeline plot
    ax2.set_xticks(range(len(experiments)))
    ax2.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(emotions)
    ax2.set_title('Training Lens Timeline: Emotion Focus Intensity', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Experiments')
    ax2.set_ylabel('Emotions')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=emotion) 
                      for emotion, color in emotion_colors.items()]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_focus_summary_table(experiments, output_path):
    """Create a summary table of emotion focus changes."""
    
    summary_data = []
    
    for exp_dir in experiments:
        config = load_experiment_config(exp_dir)
        if config:
            exp_name = os.path.basename(exp_dir).split('_')[0]
            description = config.get('experiment_description', 'N/A')
            weights = config.get('sampling_weights', {})
            
            # Find promoted and demoted emotions
            promoted = [f"{emotion} ({weight}x)" for emotion, weight in weights.items() if weight > 1.2]
            demoted = [f"{emotion} ({weight}x)" for emotion, weight in weights.items() if weight < 0.8]
            
            summary_data.append({
                'Experiment': exp_name,
                'Description': description,
                'Promoted Emotions': ', '.join(promoted) if promoted else 'None',
                'Demoted Emotions': ', '.join(demoted) if demoted else 'None',
                'Resume From': config.get('resume_from_checkpoint', 'New experiment')
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Visualize training lens progression')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Paths to experiment directories to analyze')
    parser.add_argument('--output-dir', default='./analysis',
                       help='Output directory for results')
    parser.add_argument('--name', default='training_lens',
                       help='Base name for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    plot_path = create_lens_progression_plot(
        args.experiments,
        os.path.join(args.output_dir, f"{args.name}_progression.png")
    )
    
    table_path = create_focus_summary_table(
        args.experiments,
        os.path.join(args.output_dir, f"{args.name}_summary.csv")
    )
    
    print(f"✅ Training lens analysis complete!")
    print(f"📊 Progression plot: {plot_path}")
    print(f"📋 Summary table: {table_path}")
    
    # Print summary to console
    print(f"\n📋 Training Lens Summary:")
    print("=" * 60)
    
    for exp_dir in args.experiments:
        config = load_experiment_config(exp_dir)
        if config:
            exp_name = os.path.basename(exp_dir).split('_')[0]
            description = config.get('experiment_description', 'N/A')
            weights = config.get('sampling_weights', {})
            
            print(f"\n🔬 {exp_name}:")
            print(f"   📝 {description}")
            
            # Show emotion focus
            for emotion, weight in weights.items():
                if weight != 1.0:
                    focus_type = "🔺 Promoted" if weight > 1.0 else "🔻 Demoted"
                    print(f"   {focus_type} {emotion}: {weight}x")

if __name__ == "__main__":
    main()
