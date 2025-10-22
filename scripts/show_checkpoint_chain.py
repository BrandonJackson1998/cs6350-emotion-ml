#!/usr/bin/env python3
"""
Checkpoint Chain Visualizer

This script shows the checkpoint chain and training progression,
making it easy to see how experiments were connected and what changed.
"""

import argparse
import json
import os
from pathlib import Path

def load_experiment_config(experiment_dir):
    """Load experiment configuration."""
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def find_checkpoint_chain(experiments):
    """Find the checkpoint chain starting from the first experiment."""
    
    chain = []
    current_experiment = experiments[0] if experiments else None
    
    while current_experiment and os.path.exists(current_experiment):
        config = load_experiment_config(current_experiment)
        if not config:
            break
            
        # Add to chain
        chain.append({
            'directory': current_experiment,
            'config': config
        })
        
        # Find next experiment that resumes from this one
        next_experiment = None
        for exp_dir in experiments:
            if exp_dir != current_experiment:
                exp_config = load_experiment_config(exp_dir)
                if exp_config and exp_config.get('resume_from_checkpoint'):
                    # Check if this experiment resumes from current one
                    resume_path = exp_config['resume_from_checkpoint']
                    if current_experiment in resume_path:
                        next_experiment = exp_dir
                        break
        
        current_experiment = next_experiment
    
    return chain

def print_checkpoint_chain(chain):
    """Print a visual representation of the checkpoint chain."""
    
    print("🔗 Checkpoint Chain Analysis")
    print("=" * 60)
    
    for i, exp in enumerate(chain):
        config = exp['config']
        exp_name = os.path.basename(exp['directory']).split('_')[0]
        
        print(f"\n📊 Step {i+1}: {exp_name}")
        print(f"   📁 Directory: {exp['directory']}")
        print(f"   📝 Description: {config.get('experiment_description', 'N/A')}")
        print(f"   🔢 Epochs: {config.get('num_epochs', 'N/A')}")
        
        # Show emotion focus
        weights = config.get('sampling_weights', {})
        promoted = [f"{emotion} ({weight}x)" for emotion, weight in weights.items() if weight > 1.2]
        demoted = [f"{emotion} ({weight}x)" for emotion, weight in weights.items() if weight < 0.8]
        
        if promoted:
            print(f"   🔺 Promoted: {', '.join(promoted)}")
        if demoted:
            print(f"   🔻 Demoted: {', '.join(demoted)}")
        
        # Show resume info
        resume_from = config.get('resume_from_checkpoint')
        if resume_from:
            print(f"   🔄 Resumed from: {resume_from}")
        else:
            print(f"   🚀 New experiment")
        
        # Show connection to next step
        if i < len(chain) - 1:
            print(f"   ⬇️  Connected to: Step {i+2}")
        else:
            print(f"   🏁 End of chain")

def create_chain_diagram(chain, output_path):
    """Create a visual diagram of the checkpoint chain."""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for different steps
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    
    for i, exp in enumerate(chain):
        config = exp['config']
        exp_name = os.path.basename(exp['directory']).split('_')[0]
        
        # Position
        x = i * 2
        y = 0
        
        # Draw experiment box
        rect = patches.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                               facecolor=colors[i % len(colors)], alpha=0.7, 
                               edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add experiment name
        ax.text(x, y+0.1, exp_name, ha='center', va='center', 
               fontweight='bold', fontsize=10)
        
        # Add epoch count
        epochs = config.get('num_epochs', 'N/A')
        ax.text(x, y-0.1, f"{epochs} epochs", ha='center', va='center', 
               fontsize=8)
        
        # Add emotion focus summary
        weights = config.get('sampling_weights', {})
        focus_summary = []
        for emotion, weight in weights.items():
            if weight > 1.2:
                focus_summary.append(f"{emotion}↑")
            elif weight < 0.8:
                focus_summary.append(f"{emotion}↓")
        
        if focus_summary:
            ax.text(x, y-0.25, ', '.join(focus_summary), ha='center', va='center', 
                   fontsize=7, style='italic')
        
        # Draw arrow to next step
        if i < len(chain) - 1:
            ax.arrow(x + 0.8, y, 0.4, 0, head_width=0.1, head_length=0.1, 
                    fc='black', ec='black')
    
    # Customize plot
    ax.set_xlim(-1, len(chain) * 2 - 1)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Checkpoint Chain: Training Progression', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=f'Step {i+1}')
        for i in range(len(chain))
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Visualize checkpoint chain')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Paths to experiment directories to analyze')
    parser.add_argument('--output-dir', default='./analysis',
                       help='Output directory for results')
    parser.add_argument('--name', default='checkpoint_chain',
                       help='Base name for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find checkpoint chain
    chain = find_checkpoint_chain(args.experiments)
    
    if not chain:
        print("❌ No checkpoint chain found!")
        return
    
    # Print chain analysis
    print_checkpoint_chain(chain)
    
    # Create visual diagram
    diagram_path = create_chain_diagram(
        chain,
        os.path.join(args.output_dir, f"{args.name}_diagram.png")
    )
    
    print(f"\n✅ Checkpoint chain analysis complete!")
    print(f"📊 Diagram: {diagram_path}")
    
    # Summary statistics
    print(f"\n📈 Chain Summary:")
    print(f"   🔗 Total steps: {len(chain)}")
    print(f"   🔄 Resumed experiments: {len([exp for exp in chain if exp['config'].get('resume_from_checkpoint')])}")
    print(f"   🚀 New experiments: {len([exp for exp in chain if not exp['config'].get('resume_from_checkpoint')])}")

if __name__ == "__main__":
    main()
