#!/usr/bin/env python3
"""
Temporal Emotion Analysis System
Analyzes emotion changes across sequential video frames to track emotional transitions over time.

This script processes sequential frames from /compare/test/ directory, predicts emotions for each frame,
calculates frame-to-frame emotion changes, and generates temporal visualizations.

Author: Generated for cs6350-emotion-ml project
Date: 2025-10-22
"""

import os
import sys
import glob
import re
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our existing components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark import device

# Try to import transformers components
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    print("Warning: transformers not available. Please install with 'pip install transformers'")
    AutoImageProcessor = None
    AutoModelForImageClassification = None

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class TemporalEmotionAnalyzer:
    """
    Main class for temporal emotion analysis across video frames
    """
    
    def __init__(self, model_path: str = None, model_name: str = None, 
                 emotion_labels: List[str] = None, config: Dict = None):
        """
        Initialize the temporal emotion analyzer
        
        Args:
            model_path: Path to trained model file (.pt)
            model_name: Hugging Face model name for pre-trained model
            emotion_labels: List of emotion labels in correct order
            config: Configuration dictionary
        """
        self.device = device
        self.config = config or {}
        
        # Default emotion labels (corrected order for the model)
        self.emotion_labels = emotion_labels or [
            'sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy'
        ]
        
        # Initialize model components
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.model_name = model_name or "dima806/facial_emotions_image_detection"
        
        # Data storage
        self.frame_data = []
        self.emotion_predictions = []
        self.emotion_deltas = []
        
        print(f"🔧 Initializing Temporal Emotion Analyzer")
        print(f"   Device: {self.device}")
        print(f"   Emotion Labels: {self.emotion_labels}")
        
    def load_model(self):
        """Load the emotion recognition model"""
        print(f"\n📥 Loading emotion recognition model...")
        
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom trained model
                print(f"   Loading trained model from: {self.model_path}")
                
                # Load processor from pre-trained model
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                
                # Load model architecture
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.emotion_labels),
                    ignore_mismatched_sizes=True
                )
                
                # Load trained weights
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"   ✓ Loaded custom trained model")
                
            else:
                # Load pre-trained model
                print(f"   Loading pre-trained model: {self.model_name}")
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.emotion_labels),
                    ignore_mismatched_sizes=True
                )
                print(f"   ✓ Loaded pre-trained model")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"   ✓ Model ready for inference")
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise
    
    def load_sequential_frames(self, directory_path: str) -> List[Dict]:
        """
        Load sequential frames from directory
        
        Args:
            directory_path: Path to directory containing sequential frames
            
        Returns:
            List of frame info dictionaries
        """
        print(f"\n📂 Loading sequential frames from: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory_path, ext)
            image_files.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern = os.path.join(directory_path, ext.upper())
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No image files found in {directory_path}")
        
        # Extract frame numbers and sort
        frame_info = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Extract number from filename (assuming format like "0.jpg", "1.jpg", etc.)
            match = re.search(r'(\d+)', filename)
            if match:
                frame_num = int(match.group(1))
                frame_info.append({
                    'frame_number': frame_num,
                    'filename': filename,
                    'path': img_path
                })
        
        # Sort by frame number
        frame_info.sort(key=lambda x: x['frame_number'])
        
        print(f"   ✓ Found {len(frame_info)} sequential frames")
        if frame_info:
            print(f"   Frame range: {frame_info[0]['frame_number']} to {frame_info[-1]['frame_number']}")
        
        # Check for gaps in sequence
        if len(frame_info) > 1:
            expected_frames = set(range(frame_info[0]['frame_number'], frame_info[-1]['frame_number'] + 1))
            actual_frames = set(f['frame_number'] for f in frame_info)
            missing_frames = expected_frames - actual_frames
            
            if missing_frames:
                print(f"   ⚠️  Missing frames: {sorted(missing_frames)}")
                if self.config.get('strict_sequence', False):
                    raise ValueError(f"Missing frames detected: {sorted(missing_frames)}")
        
        self.frame_data = frame_info
        return frame_info
    
    def predict_frame_emotions(self, frame_list: List[Dict]) -> np.ndarray:
        """
        Predict emotions for all frames
        
        Args:
            frame_list: List of frame information dictionaries
            
        Returns:
            Numpy array of emotion probabilities (n_frames, n_emotions)
        """
        print(f"\n🔮 Predicting emotions for {len(frame_list)} frames...")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        batch_size = self.config.get('batch_size', 8)
        all_predictions = []
        
        # Process frames in batches
        for i in tqdm(range(0, len(frame_list), batch_size), desc="Processing frames"):
            batch = frame_list[i:i + batch_size]
            batch_predictions = self._process_frame_batch(batch)
            all_predictions.extend(batch_predictions)
        
        # Convert to numpy array
        emotion_probabilities = np.array(all_predictions)
        
        print(f"   ✓ Generated emotion predictions: {emotion_probabilities.shape}")
        print(f"   Probability range: [{emotion_probabilities.min():.3f}, {emotion_probabilities.max():.3f}]")
        
        self.emotion_predictions = emotion_probabilities
        return emotion_probabilities
    
    def _process_frame_batch(self, batch: List[Dict]) -> List[np.ndarray]:
        """Process a batch of frames"""
        batch_images = []
        batch_predictions = []
        
        # Load and preprocess images
        for frame_info in batch:
            try:
                image = Image.open(frame_info['path']).convert('RGB')
                batch_images.append(image)
            except Exception as e:
                print(f"   ⚠️  Failed to load frame {frame_info['frame_number']}: {e}")
                # Use a black image as fallback
                batch_images.append(Image.new('RGB', (224, 224), color='black'))
        
        # Process batch
        if batch_images:
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    batch_predictions = probabilities.cpu().numpy()
                    
            except Exception as e:
                print(f"   ⚠️  Batch processing error: {e}")
                # Return zero probabilities as fallback
                batch_predictions = np.zeros((len(batch_images), len(self.emotion_labels)))
        
        return batch_predictions
    
    def calculate_emotion_deltas(self, emotion_probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate frame-to-frame emotion changes
        
        Args:
            emotion_probabilities: Array of emotion probabilities (n_frames, n_emotions)
            
        Returns:
            Array of emotion deltas (n_frames-1, n_emotions)
        """
        print(f"\n📊 Calculating emotion deltas...")
        
        if len(emotion_probabilities) < 2:
            print("   ⚠️  Need at least 2 frames to calculate deltas")
            return np.array([])
        
        # Calculate frame-to-frame differences
        deltas = np.diff(emotion_probabilities, axis=0)
        
        # Apply smoothing if configured
        smoothing_window = self.config.get('smoothing_window', 1)
        if smoothing_window > 1:
            print(f"   Applying smoothing with window size: {smoothing_window}")
            # Apply moving average smoothing to deltas
            deltas = self._apply_smoothing(deltas, smoothing_window)
        
        print(f"   ✓ Calculated deltas: {deltas.shape}")
        print(f"   Delta range: [{deltas.min():.4f}, {deltas.max():.4f}]")
        
        # Calculate some statistics
        print(f"   Mean absolute change: {np.mean(np.abs(deltas)):.4f}")
        print(f"   Max change: {np.max(np.abs(deltas)):.4f}")
        
        self.emotion_deltas = deltas
        return deltas
    
    def _apply_smoothing(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average smoothing"""
        if window_size <= 1:
            return data
        
        # Pad data for boundary handling
        pad_size = window_size // 2
        padded_data = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
        
        # Apply moving average
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = i
            end_idx = i + window_size
            smoothed[i] = np.mean(padded_data[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def generate_temporal_plots(self, output_dir: str):
        """
        Generate temporal visualization plots
        
        Args:
            output_dir: Directory to save plots
        """
        print(f"\n📈 Generating temporal plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.emotion_predictions) == 0:
            print("   ⚠️  No emotion predictions available for plotting")
            return
        
        # Create frame indices
        frame_indices = range(len(self.emotion_predictions))
        
        # Plot 1: Emotion probabilities over time
        self._plot_emotion_probabilities(frame_indices, output_dir)
        
        # Plot 2: Emotion changes over time
        if len(self.emotion_deltas) > 0:
            self._plot_emotion_deltas(frame_indices, output_dir)
        
        # Plot 3: Summary statistics
        self._plot_emotion_statistics(output_dir)
        
        print(f"   ✓ Plots saved to: {output_dir}")
    
    def _plot_emotion_probabilities(self, frame_indices: range, output_dir: str):
        """Plot emotion probabilities over time"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots for better visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: All emotions on one plot
        for i, emotion in enumerate(self.emotion_labels):
            ax1.plot(frame_indices, self.emotion_predictions[:, i], 
                    label=emotion, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Emotion Probability')
        ax1.set_title('Emotion Probabilities Over Time', fontsize=16, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Heatmap view
        im = ax2.imshow(self.emotion_predictions.T, aspect='auto', cmap='viridis', 
                       interpolation='nearest')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Emotions')
        ax2.set_title('Emotion Probability Heatmap', fontsize=16, fontweight='bold')
        ax2.set_yticks(range(len(self.emotion_labels)))
        ax2.set_yticklabels(self.emotion_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Probability')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_probabilities_temporal.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emotion_deltas(self, frame_indices: range, output_dir: str):
        """Plot emotion changes over time"""
        if len(self.emotion_deltas) == 0:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Delta frame indices (one less than total frames)
        delta_indices = range(1, len(frame_indices))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Delta curves
        for i, emotion in enumerate(self.emotion_labels):
            ax1.plot(delta_indices, self.emotion_deltas[:, i], 
                    label=emotion, linewidth=2, alpha=0.8)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Frame Transition')
        ax1.set_ylabel('Emotion Probability Change')
        ax1.set_title('Emotion Changes Between Consecutive Frames', fontsize=16, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Absolute changes
        abs_deltas = np.abs(self.emotion_deltas)
        for i, emotion in enumerate(self.emotion_labels):
            ax2.plot(delta_indices, abs_deltas[:, i], 
                    label=emotion, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Frame Transition')
        ax2.set_ylabel('Absolute Emotion Change')
        ax2.set_title('Absolute Emotion Changes (Volatility)', fontsize=16, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_deltas_temporal.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emotion_statistics(self, output_dir: str):
        """Plot summary statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average emotion probabilities
        avg_emotions = np.mean(self.emotion_predictions, axis=0)
        bars1 = ax1.bar(self.emotion_labels, avg_emotions, alpha=0.7)
        ax1.set_ylabel('Average Probability')
        ax1.set_title('Average Emotion Probabilities', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars1, avg_emotions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 2: Emotion volatility (std of probabilities)
        emotion_std = np.std(self.emotion_predictions, axis=0)
        bars2 = ax2.bar(self.emotion_labels, emotion_std, alpha=0.7, color='orange')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Emotion Volatility', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars2, emotion_std):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 3: Total absolute changes
        if len(self.emotion_deltas) > 0:
            total_abs_changes = np.sum(np.abs(self.emotion_deltas), axis=0)
            bars3 = ax3.bar(self.emotion_labels, total_abs_changes, alpha=0.7, color='green')
            ax3.set_ylabel('Total Absolute Change')
            ax3.set_title('Total Emotion Changes', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars3, total_abs_changes):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 4: Max absolute changes
        if len(self.emotion_deltas) > 0:
            max_abs_changes = np.max(np.abs(self.emotion_deltas), axis=0)
            bars4 = ax4.bar(self.emotion_labels, max_abs_changes, alpha=0.7, color='red')
            ax4.set_ylabel('Maximum Absolute Change')
            ax4.set_title('Peak Emotion Changes', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars4, max_abs_changes):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_statistics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_data(self, output_dir: str):
        """
        Save emotion data and analysis results
        
        Args:
            output_dir: Directory to save data files
        """
        print(f"\n💾 Saving analysis data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save emotion probabilities
        if len(self.emotion_predictions) > 0:
            prob_df = pd.DataFrame(self.emotion_predictions, columns=self.emotion_labels)
            prob_df.index.name = 'frame_number'
            prob_df.to_csv(os.path.join(output_dir, 'emotion_probabilities.csv'))
            print(f"   ✓ Saved emotion probabilities: emotion_probabilities.csv")
        
        # Save emotion deltas
        if len(self.emotion_deltas) > 0:
            delta_df = pd.DataFrame(self.emotion_deltas, columns=self.emotion_labels)
            delta_df.index.name = 'transition_number'
            delta_df.to_csv(os.path.join(output_dir, 'emotion_deltas.csv'))
            print(f"   ✓ Saved emotion deltas: emotion_deltas.csv")
        
        # Save frame information
        if self.frame_data:
            frame_df = pd.DataFrame(self.frame_data)
            frame_df.to_csv(os.path.join(output_dir, 'frame_information.csv'), index=False)
            print(f"   ✓ Saved frame information: frame_information.csv")
        
        # Save summary statistics
        summary = self._calculate_summary_statistics()
        with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✓ Saved analysis summary: analysis_summary.json")
        
        # Save configuration
        config_data = {
            'model_path': self.model_path,
            'model_name': self.model_name,
            'emotion_labels': self.emotion_labels,
            'config': self.config,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_frames': len(self.emotion_predictions),
            'total_transitions': len(self.emotion_deltas)
        }
        
        with open(os.path.join(output_dir, 'analysis_config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"   ✓ Saved configuration: analysis_config.json")
    
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate comprehensive summary statistics"""
        summary = {
            'total_frames': len(self.emotion_predictions),
            'total_transitions': len(self.emotion_deltas),
            'emotion_labels': self.emotion_labels
        }
        
        if len(self.emotion_predictions) > 0:
            summary['emotion_statistics'] = {}
            for i, emotion in enumerate(self.emotion_labels):
                probs = self.emotion_predictions[:, i]
                summary['emotion_statistics'][emotion] = {
                    'mean_probability': float(np.mean(probs)),
                    'std_probability': float(np.std(probs)),
                    'min_probability': float(np.min(probs)),
                    'max_probability': float(np.max(probs)),
                    'median_probability': float(np.median(probs))
                }
        
        if len(self.emotion_deltas) > 0:
            summary['change_statistics'] = {}
            for i, emotion in enumerate(self.emotion_labels):
                deltas = self.emotion_deltas[:, i]
                abs_deltas = np.abs(deltas)
                summary['change_statistics'][emotion] = {
                    'total_absolute_change': float(np.sum(abs_deltas)),
                    'mean_absolute_change': float(np.mean(abs_deltas)),
                    'max_absolute_change': float(np.max(abs_deltas)),
                    'std_change': float(np.std(deltas)),
                    'positive_changes': int(np.sum(deltas > 0)),
                    'negative_changes': int(np.sum(deltas < 0))
                }
        
        return summary
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns in emotion changes
        
        Returns:
            Dictionary with pattern analysis results
        """
        print(f"\n🔍 Analyzing temporal patterns...")
        
        if len(self.emotion_deltas) == 0:
            print("   ⚠️  No deltas available for pattern analysis")
            return {}
        
        patterns = {}
        
        # Find significant changes
        threshold = self.config.get('change_threshold', 0.1)
        significant_changes = np.abs(self.emotion_deltas) > threshold
        
        for i, emotion in enumerate(self.emotion_labels):
            emotion_patterns = {
                'significant_changes': int(np.sum(significant_changes[:, i])),
                'change_frequency': float(np.mean(significant_changes[:, i])),
                'largest_increase': float(np.max(self.emotion_deltas[:, i])),
                'largest_decrease': float(np.min(self.emotion_deltas[:, i]))
            }
            
            # Find frames with largest changes
            if len(self.emotion_deltas) > 0:
                max_change_idx = np.argmax(np.abs(self.emotion_deltas[:, i]))
                emotion_patterns['max_change_frame'] = int(max_change_idx + 1)  # +1 because deltas start from frame 1
                emotion_patterns['max_change_value'] = float(self.emotion_deltas[max_change_idx, i])
            
            patterns[emotion] = emotion_patterns
        
        print(f"   ✓ Pattern analysis complete")
        return patterns


def create_default_config() -> Dict:
    """Create default configuration"""
    return {
        'batch_size': 8,
        'smoothing_window': 1,
        'change_threshold': 0.1,
        'strict_sequence': False,
        'plot_style': 'default',
        'output_format': 'png',
        'dpi': 300
    }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Temporal Emotion Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with pre-trained model
    python temporal_emotion_analyzer.py /compare/test/
    
    # Use custom trained model
    python temporal_emotion_analyzer.py /compare/test/ --model-path /path/to/model.pt
    
    # Apply smoothing and save to custom directory
    python temporal_emotion_analyzer.py /compare/test/ --smoothing 3 --output /custom/output/
        """
    )
    
    parser.add_argument('input_dir', help='Directory containing sequential frames')
    parser.add_argument('--model-path', help='Path to trained model file (.pt)')
    parser.add_argument('--model-name', default='dima806/facial_emotions_image_detection',
                       help='Hugging Face model name (default: dima806/facial_emotions_image_detection)')
    parser.add_argument('--output', default='./compare/outputs/temporal_analysis',
                       help='Output directory (default: ./compare/outputs/temporal_analysis)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--smoothing', type=int, default=1,
                       help='Smoothing window size (default: 1, no smoothing)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Change significance threshold (default: 0.1)')
    parser.add_argument('--strict', action='store_true',
                       help='Require strict sequential frames (no gaps)')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_default_config()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Update config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'smoothing_window': args.smoothing,
        'change_threshold': args.threshold,
        'strict_sequence': args.strict
    })
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output}_{timestamp}"
    
    print(f"🚀 Starting Temporal Emotion Analysis")
    print(f"=" * 60)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model: {args.model_name}")
    if args.model_path:
        print(f"Custom Model: {args.model_path}")
    print(f"Configuration: {config}")
    print(f"=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = TemporalEmotionAnalyzer(
            model_path=args.model_path,
            model_name=args.model_name,
            config=config
        )
        
        # Load model
        analyzer.load_model()
        
        # Load frames
        frames = analyzer.load_sequential_frames(args.input_dir)
        
        if len(frames) < 2:
            print("❌ Need at least 2 frames for temporal analysis")
            return
        
        # Predict emotions
        emotion_probs = analyzer.predict_frame_emotions(frames)
        
        # Calculate deltas
        emotion_deltas = analyzer.calculate_emotion_deltas(emotion_probs)
        
        # Generate visualizations
        analyzer.generate_temporal_plots(output_dir)
        
        # Save data
        analyzer.save_data(output_dir)
        
        # Analyze patterns
        patterns = analyzer.analyze_temporal_patterns()
        
        # Save patterns
        if patterns:
            with open(os.path.join(output_dir, 'temporal_patterns.json'), 'w') as f:
                json.dump(patterns, f, indent=2)
            print(f"   ✓ Saved temporal patterns: temporal_patterns.json")
        
        print(f"\n🎉 Analysis Complete!")
        print(f"=" * 60)
        print(f"📊 Results Summary:")
        print(f"   Frames processed: {len(frames)}")
        print(f"   Transitions analyzed: {len(emotion_deltas)}")
        print(f"   Output directory: {output_dir}")
        print(f"\n📁 Generated files:")
        print(f"   • emotion_probabilities.csv - Frame-by-frame emotion probabilities")
        print(f"   • emotion_deltas.csv - Frame-to-frame emotion changes")
        print(f"   • emotion_probabilities_temporal.png - Temporal emotion plots")
        print(f"   • emotion_deltas_temporal.png - Emotion change plots")
        print(f"   • emotion_statistics.png - Summary statistics plots")
        print(f"   • analysis_summary.json - Comprehensive analysis summary")
        print(f"   • temporal_patterns.json - Pattern analysis results")
        print(f"=" * 60)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())