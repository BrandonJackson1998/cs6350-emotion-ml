#!/usr/bin/env python3
"""
Data Validation Script for Emotion ML Dataset
Analyzes image dimensions, file sizes, pixel values, and data quality
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.results = {
            'summary': {},
            'file_analysis': [],
            'class_distribution': {},
            'dimension_stats': {},
            'pixel_stats': {},
            'quality_issues': [],
            'grayscale_analysis': {}
        }
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def analyze_image(self, image_path):
        """Analyze a single image and return comprehensive metrics"""
        try:
            # File size
            file_size = os.path.getsize(image_path)
            
            # Load image
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Convert to numpy array for pixel analysis
                img_array = np.array(img)
                
                # Basic pixel statistics
                pixel_min = img_array.min()
                pixel_max = img_array.max()
                pixel_mean = img_array.mean()
                pixel_std = img_array.std()
                
                # Check if grayscale
                is_grayscale = len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1)
                
                # For color images, analyze RGB channels
                rgb_stats = {}
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    for i, channel in enumerate(['R', 'G', 'B']):
                        channel_data = img_array[:, :, i]
                        rgb_stats[channel] = {
                            'min': channel_data.min(),
                            'max': channel_data.max(),
                            'mean': channel_data.mean(),
                            'std': channel_data.std()
                        }
                    
                    # Check if image is actually grayscale (all channels identical)
                    r_channel = img_array[:, :, 0]
                    g_channel = img_array[:, :, 1]
                    b_channel = img_array[:, :, 2]
                    is_actually_grayscale = np.array_equal(r_channel, g_channel) and np.array_equal(g_channel, b_channel)
                else:
                    is_actually_grayscale = is_grayscale
                
                return {
                    'file_path': str(image_path),
                    'file_size_bytes': file_size,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height,
                    'total_pixels': width * height,
                    'pixel_min': pixel_min,
                    'pixel_max': pixel_max,
                    'pixel_mean': pixel_mean,
                    'pixel_std': pixel_std,
                    'is_grayscale_format': is_grayscale,
                    'is_actually_grayscale': is_actually_grayscale,
                    'rgb_stats': rgb_stats,
                    'load_success': True,
                    'error': None
                }
                
        except Exception as e:
            return {
                'file_path': str(image_path),
                'load_success': False,
                'error': str(e),
                'file_size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
            }
    
    def analyze_dataset(self):
        """Analyze the entire dataset"""
        print("🔍 Starting dataset validation...")
        
        total_images = 0
        successful_loads = 0
        failed_loads = 0
        
        # Initialize counters
        class_counts = defaultdict(int)
        dimension_counter = Counter()
        file_size_stats = []
        pixel_value_ranges = []
        grayscale_count = 0
        color_count = 0
        
        # Analyze train and test sets
        for split in ['train', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                print(f"⚠️  {split} directory not found")
                continue
                
            print(f"📁 Analyzing {split} set...")
            
            for emotion in self.emotions:
                emotion_dir = split_dir / emotion
                if not emotion_dir.exists():
                    print(f"⚠️  {emotion} directory not found in {split}")
                    continue
                
                image_files = list(emotion_dir.glob('*.jpg'))
                print(f"  📊 {emotion}: {len(image_files)} images")
                
                for img_path in image_files:
                    total_images += 1
                    analysis = self.analyze_image(img_path)
                    
                    # Store detailed analysis
                    analysis['split'] = split
                    analysis['emotion'] = emotion
                    self.results['file_analysis'].append(analysis)
                    
                    if analysis['load_success']:
                        successful_loads += 1
                        class_counts[f"{split}_{emotion}"] += 1
                        
                        # Collect statistics
                        dimension_counter[(analysis['width'], analysis['height'])] += 1
                        file_size_stats.append(analysis['file_size_bytes'])
                        pixel_value_ranges.append((analysis['pixel_min'], analysis['pixel_max']))
                        
                        # Grayscale analysis
                        if analysis['is_actually_grayscale']:
                            grayscale_count += 1
                        else:
                            color_count += 1
                    else:
                        failed_loads += 1
                        self.results['quality_issues'].append({
                            'file': str(img_path),
                            'issue': 'Failed to load',
                            'error': analysis['error']
                        })
        
        # Calculate summary statistics
        self.results['summary'] = {
            'total_images': total_images,
            'successful_loads': successful_loads,
            'failed_loads': failed_loads,
            'success_rate': successful_loads / total_images if total_images > 0 else 0,
            'grayscale_images': grayscale_count,
            'color_images': color_count,
            'grayscale_percentage': (grayscale_count / successful_loads * 100) if successful_loads > 0 else 0
        }
        
        # Class distribution
        self.results['class_distribution'] = dict(class_counts)
        
        # Dimension analysis
        if dimension_counter:
            dimensions = list(dimension_counter.keys())
            widths = [d[0] for d in dimensions]
            heights = [d[1] for d in dimensions]
            
            self.results['dimension_stats'] = {
                'unique_dimensions': len(dimension_counter),
                'most_common_dimension': dimension_counter.most_common(1)[0] if dimension_counter else None,
                'width_stats': {
                    'min': min(widths),
                    'max': max(widths),
                    'mean': np.mean(widths),
                    'std': np.std(widths)
                },
                'height_stats': {
                    'min': min(heights),
                    'max': max(heights),
                    'mean': np.mean(heights),
                    'std': np.std(heights)
                },
                'all_dimensions': dict(dimension_counter)
            }
        
        # File size analysis
        if file_size_stats:
            self.results['file_size_stats'] = {
                'min_bytes': min(file_size_stats),
                'max_bytes': max(file_size_stats),
                'mean_bytes': np.mean(file_size_stats),
                'std_bytes': np.std(file_size_stats),
                'median_bytes': np.median(file_size_stats)
            }
        
        # Pixel value analysis
        if pixel_value_ranges:
            pixel_mins = [r[0] for r in pixel_value_ranges]
            pixel_maxs = [r[1] for r in pixel_value_ranges]
            
            self.results['pixel_stats'] = {
                'min_pixel_value': min(pixel_mins),
                'max_pixel_value': max(pixel_maxs),
                'mean_min': np.mean(pixel_mins),
                'mean_max': np.mean(pixel_maxs),
                'normalized_range': f"{min(pixel_mins)}-{max(pixel_maxs)}"
            }
        
        print(f"✅ Analysis complete: {successful_loads}/{total_images} images processed successfully")
        
    def generate_report(self):
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Data Validation Report
Generated: {timestamp}

## Executive Summary
- **Total Images**: {self.results['summary']['total_images']:,}
- **Success Rate**: {self.results['summary']['success_rate']:.2%}
- **Failed Loads**: {self.results['summary']['failed_loads']}
- **Grayscale Images**: {self.results['summary']['grayscale_images']:,} ({self.results['summary']['grayscale_percentage']:.1f}%)
- **Color Images**: {self.results['summary']['color_images']:,}

## Dataset Structure
"""
        
        # Class distribution
        report += "\n### Class Distribution\n"
        for split in ['train', 'test']:
            report += f"\n#### {split.title()} Set\n"
            for emotion in self.emotions:
                count = self.results['class_distribution'].get(f"{split}_{emotion}", 0)
                report += f"- **{emotion.title()}**: {count:,} images\n"
        
        # Dimension analysis
        if 'dimension_stats' in self.results:
            dim_stats = self.results['dimension_stats']
            report += f"""
## Image Dimensions
- **Unique Dimensions**: {dim_stats['unique_dimensions']}
- **Most Common**: {dim_stats['most_common_dimension'][0]} ({dim_stats['most_common_dimension'][1]} images)
- **Width Range**: {dim_stats['width_stats']['min']} - {dim_stats['width_stats']['max']} (mean: {dim_stats['width_stats']['mean']:.1f})
- **Height Range**: {dim_stats['height_stats']['min']} - {dim_stats['height_stats']['max']} (mean: {dim_stats['height_stats']['mean']:.1f})
"""
        
        # File size analysis
        if 'file_size_stats' in self.results:
            size_stats = self.results['file_size_stats']
            report += f"""
## File Size Analysis
- **Size Range**: {size_stats['min_bytes']:,} - {size_stats['max_bytes']:,} bytes
- **Mean Size**: {size_stats['mean_bytes']:,.0f} bytes
- **Median Size**: {size_stats['median_bytes']:,.0f} bytes
- **Std Deviation**: {size_stats['std_bytes']:,.0f} bytes
"""
        
        # Pixel value analysis
        if 'pixel_stats' in self.results:
            pixel_stats = self.results['pixel_stats']
            report += f"""
## Pixel Value Analysis
- **Value Range**: {pixel_stats['min_pixel_value']} - {pixel_stats['max_pixel_value']}
- **Mean Min Value**: {pixel_stats['mean_min']:.2f}
- **Mean Max Value**: {pixel_stats['mean_max']:.2f}
- **Normalization Status**: {'Normalized (0-1)' if pixel_stats['max_pixel_value'] <= 1 else 'Raw values (0-255)'}
"""
        
        # Quality issues
        if self.results['quality_issues']:
            report += f"""
## Quality Issues Found
- **Total Issues**: {len(self.results['quality_issues'])}
"""
            for issue in self.results['quality_issues'][:10]:  # Show first 10
                report += f"- {issue['file']}: {issue['issue']}\n"
            if len(self.results['quality_issues']) > 10:
                report += f"- ... and {len(self.results['quality_issues']) - 10} more issues\n"
        
        # Recommendations
        report += """
## Recommendations
"""
        
        # Check for normalization
        if 'pixel_stats' in self.results:
            if self.results['pixel_stats']['max_pixel_value'] > 1:
                report += "- ⚠️  **Pixel values are not normalized** (0-255 range detected)\n"
            else:
                report += "- ✅ **Pixel values appear normalized** (0-1 range)\n"
        
        # Check for dimension consistency
        if 'dimension_stats' in self.results:
            if self.results['dimension_stats']['unique_dimensions'] > 1:
                report += "- ⚠️  **Inconsistent image dimensions** detected\n"
            else:
                report += "- ✅ **Consistent image dimensions** across dataset\n"
        
        # Check for class balance
        train_counts = [self.results['class_distribution'].get(f"train_{emotion}", 0) for emotion in self.emotions]
        if train_counts:
            max_count = max(train_counts)
            min_count = min(train_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            if imbalance_ratio > 2:
                report += f"- ⚠️  **Class imbalance detected** (ratio: {imbalance_ratio:.1f}:1)\n"
            else:
                report += "- ✅ **Classes are reasonably balanced**\n"
        
        return report
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(k): self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_results(self):
        """Save results to files"""
        # Save JSON results (convert tuple keys to strings)
        json_results = self._convert_for_json(self.results)
        with open(self.data_dir / 'validation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save CSV for detailed analysis
        if self.results['file_analysis']:
            with open(self.data_dir / 'image_analysis.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results['file_analysis'][0].keys())
                writer.writeheader()
                writer.writerows(self.results['file_analysis'])
        
        # Save markdown report
        report = self.generate_report()
        with open(self.data_dir / 'data_validation_report.md', 'w') as f:
            f.write(report)
        
        print(f"📄 Results saved to:")
        print(f"  - validation_results.json")
        print(f"  - image_analysis.csv") 
        print(f"  - data_validation_report.md")

def main():
    """Main execution function"""
    data_dir = Path(__file__).parent
    validator = DataValidator(data_dir)
    
    print("🚀 Starting Emotion ML Dataset Validation")
    print("=" * 50)
    
    validator.analyze_dataset()
    validator.save_results()
    
    print("\n" + "=" * 50)
    print("✅ Validation complete!")

if __name__ == "__main__":
    main()
