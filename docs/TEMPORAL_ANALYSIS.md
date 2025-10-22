# Temporal Emotion Analysis System

## Overview

The Temporal Emotion Analysis System tracks emotional changes across sequential video frames, providing insights into how emotions evolve over time. This system is designed to analyze emotion transitions in video sequences by processing consecutive frames and calculating probability changes between frames.

## Features

### Core Capabilities
- **Sequential Frame Processing**: Automatically detects and processes numbered frames (0.jpg, 1.jpg, 2.jpg, ...)
- **Emotion Prediction**: Uses trained emotion recognition models to predict probabilities for 7 emotions
- **Change Detection**: Calculates frame-to-frame emotion probability changes (deltas)
- **Temporal Visualization**: Generates comprehensive plots showing emotion evolution over time
- **Pattern Analysis**: Identifies significant emotional transitions and patterns
- **Data Export**: Saves results in CSV, JSON, and PNG formats

### Supported Emotions
The system tracks these 7 emotions (in model-expected order):
1. **Sad** - Sadness, melancholy
2. **Disgust** - Disgust, revulsion  
3. **Angry** - Anger, irritation
4. **Neutral** - Neutral, calm expression
5. **Fear** - Fear, anxiety
6. **Surprise** - Surprise, shock
7. **Happy** - Happiness, joy

## Usage

### Basic Usage
```bash
# Analyze frames in /compare/test/ with pre-trained model
make temporal-analysis

# Or run directly
python src/temporal_emotion_analyzer.py compare/test/
```

### Advanced Usage
```bash
# Use custom trained model
make temporal-analysis-custom MODEL_PATH=experiments/best_model.pt

# Apply smoothing and custom output
python src/temporal_emotion_analyzer.py compare/test/ \
    --smoothing 3 \
    --output ./custom_analysis \
    --batch-size 16 \
    --threshold 0.15
```

### Command Line Options
- `input_dir`: Directory containing sequential frames (required)
- `--model-path`: Path to custom trained model (.pt file)
- `--model-name`: Hugging Face model name (default: dima806/facial_emotions_image_detection)
- `--output`: Output directory (default: ./outputs/temporal_analysis)
- `--batch-size`: Processing batch size (default: 8)
- `--smoothing`: Smoothing window size (default: 1, no smoothing)
- `--threshold`: Change significance threshold (default: 0.1)
- `--strict`: Require strict sequential frames (no gaps)
- `--config`: JSON configuration file
- `--verbose`: Verbose output

## Input Requirements

### Directory Structure
```
/compare/test/
├── 0.jpg    # First frame
├── 1.jpg    # Second frame  
├── 2.jpg    # Third frame
├── ...
└── N.jpg    # Last frame
```

### Frame Requirements
- **Format**: Standard image formats (JPG, PNG, BMP, TIFF)
- **Naming**: Sequential numbers (0.jpg, 1.jpg, 2.jpg, ...)
- **Content**: Face images suitable for emotion recognition
- **Minimum**: At least 2 frames required for temporal analysis

## Output Files

### Generated Files
When analysis completes, the following files are generated:

#### Data Files
- **`emotion_probabilities.csv`** - Frame-by-frame emotion probabilities
- **`emotion_deltas.csv`** - Frame-to-frame emotion changes  
- **`frame_information.csv`** - Frame metadata and paths
- **`analysis_summary.json`** - Comprehensive statistical summary
- **`temporal_patterns.json`** - Pattern analysis results
- **`analysis_config.json`** - Configuration and parameters used

#### Visualization Files
- **`emotion_probabilities_temporal.png`** - Emotion curves over time + heatmap
- **`emotion_deltas_temporal.png`** - Change plots and volatility analysis
- **`emotion_statistics.png`** - Summary statistics and comparisons

### Sample Output Structure
```
outputs/temporal_analysis_20251022_094433/
├── emotion_probabilities.csv           # Emotion data
├── emotion_deltas.csv                  # Change data
├── frame_information.csv               # Frame metadata
├── analysis_summary.json               # Statistics
├── temporal_patterns.json              # Patterns
├── analysis_config.json                # Configuration
├── emotion_probabilities_temporal.png  # Time series plots
├── emotion_deltas_temporal.png         # Change plots
└── emotion_statistics.png              # Summary plots
```

## Example Results

### Sample Analysis Output
```
🎉 Analysis Complete!
============================================================
📊 Results Summary:
   Frames processed: 3
   Transitions analyzed: 2
   Output directory: ./outputs/temporal_analysis_20251022_094433

📊 Key Findings:
   - Frame 0: 98.5% Happy (dominant emotion)
   - Frame 1: 75.9% Angry (major transition)  
   - Frame 2: 97.6% Happy (return to happiness)
   
📈 Significant Changes:
   - Frames 0→1: Happy drops 98.2%, Angry rises 75.8%
   - Frames 1→2: Angry drops 75.8%, Happy rises 97.3%
```

### Emotion Data Format
**emotion_probabilities.csv:**
```csv
frame_number,sad,disgust,angry,neutral,fear,surprise,happy
0,0.0017,0.0015,0.0014,0.0043,0.0016,0.0047,0.9848
1,0.1301,0.0035,0.7596,0.0742,0.0221,0.0073,0.0032
2,0.0016,0.0013,0.0012,0.0136,0.0017,0.0049,0.9758
```

**emotion_deltas.csv:**
```csv
transition_number,sad,disgust,angry,neutral,fear,surprise,happy
0,0.1284,0.0021,0.7581,0.0699,0.0205,0.0026,-0.9815
1,-0.1285,-0.0022,-0.7584,-0.0605,-0.0204,-0.0025,0.9725
```

## Configuration

### Default Configuration
```json
{
  "batch_size": 8,
  "smoothing_window": 1,
  "change_threshold": 0.1,
  "strict_sequence": false,
  "plot_style": "default",
  "output_format": "png",
  "dpi": 300
}
```

### Custom Configuration File
Create a JSON file with custom settings:
```json
{
  "batch_size": 16,
  "smoothing_window": 3,
  "change_threshold": 0.15,
  "strict_sequence": true,
  "plot_style": "seaborn",
  "output_format": "pdf",
  "dpi": 600
}
```

## Technical Details

### Model Integration
- **Pre-trained Model**: dima806/facial_emotions_image_detection (default)
- **Custom Models**: Support for any trained emotion recognition model
- **Model Format**: PyTorch (.pt) state dict files
- **Processing**: Batch processing for efficiency
- **Device Support**: CPU and GPU (CUDA) acceleration

### Analysis Methods

#### Delta Calculation
```python
# Frame-to-frame change calculation
deltas = probabilities[frame_i+1] - probabilities[frame_i]

# Smoothing (optional)
smoothed_deltas = moving_average(deltas, window_size)
```

#### Pattern Detection
- **Significant Changes**: Changes above threshold (default: 0.1)
- **Trend Analysis**: Increasing/decreasing emotion patterns
- **Volatility Metrics**: Standard deviation of changes
- **Peak Detection**: Largest emotion changes

#### Visualization Components
1. **Time Series Plots**: Emotion probabilities over frames
2. **Heatmap View**: Emotion intensity visualization
3. **Delta Plots**: Frame-to-frame changes
4. **Statistical Summaries**: Average, volatility, extremes

### Performance Characteristics
- **Processing Speed**: ~1-3 frames/second (CPU)
- **Memory Usage**: ~100MB per 100 frames
- **Scalability**: Handles 10-1000+ frames
- **Batch Processing**: Configurable batch sizes

## Use Cases

### Video Analysis Applications
- **Facial Expression Analysis**: Track emotional changes in video sequences
- **Reaction Studies**: Analyze emotional responses to stimuli
- **Behavioral Research**: Study emotion patterns over time
- **Content Analysis**: Evaluate emotional content in video

### Research Applications  
- **Emotion Recognition Research**: Temporal pattern analysis
- **Psychology Studies**: Emotional transition research
- **Human-Computer Interaction**: Emotion-aware systems
- **Media Analysis**: Content emotion profiling

## Troubleshooting

### Common Issues

#### Missing Frames
```
⚠️ Missing frames: [3, 5, 7]
Solution: Use --strict flag to enforce complete sequences
```

#### Memory Issues
```
CUDA out of memory
Solution: Reduce --batch-size or use CPU processing
```

#### Model Loading Errors
```
❌ Failed to load model: /path/to/model.pt
Solution: Verify model path and compatibility
```

### Performance Optimization
- **Reduce Batch Size**: Lower memory usage
- **Use GPU**: Faster processing with CUDA
- **Skip Smoothing**: Faster analysis without smoothing
- **Compress Images**: Smaller input images

## Extension and Customization

### Adding New Models
```python
# Custom model integration
analyzer = TemporalEmotionAnalyzer(
    model_path="/path/to/custom_model.pt",
    emotion_labels=["custom", "emotion", "labels"]
)
```

### Custom Analysis Metrics
```python
# Add custom analysis functions
def custom_pattern_analysis(emotion_deltas):
    # Custom pattern detection logic
    return custom_patterns
```

### Output Format Customization
```python
# Custom visualization styles
config = {
    "plot_style": "custom",
    "color_scheme": "custom_palette",
    "figure_size": (20, 12)
}
```

## Future Enhancements

### Planned Features
- **Real-time Processing**: Live video stream analysis
- **Multi-face Tracking**: Multiple faces in single frame
- **Emotion Intensity**: Calibrated emotion strength metrics
- **Advanced Smoothing**: Kalman filtering and other techniques
- **Interactive Visualization**: Web-based dashboard
- **Batch Video Processing**: Automatic video-to-frames conversion

### Integration Opportunities
- **OpenCV Integration**: Direct video file processing
- **Web Interface**: Browser-based analysis tool
- **API Endpoints**: REST API for remote analysis
- **Database Storage**: Persistent result storage
- **Cloud Processing**: Scalable cloud-based analysis

## License and Credits

This temporal emotion analysis system is part of the cs6350-emotion-ml project, utilizing:
- **Base Model**: dima806/facial_emotions_image_detection (Hugging Face)
- **Framework**: PyTorch and Transformers
- **Visualization**: Matplotlib and Seaborn
- **Data Processing**: NumPy and Pandas

Generated on: 2025-10-22
Version: 1.0.0