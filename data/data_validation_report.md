# Data Validation Report
Generated: 2025-10-07 08:53:28

## Executive Summary
- **Total Images**: 35,887
- **Success Rate**: 100.00%
- **Failed Loads**: 0
- **Grayscale Images**: 35,887 (100.0%)
- **Color Images**: 0

## Dataset Structure

### Class Distribution

#### Train Set
- **Angry**: 3,995 images
- **Disgust**: 436 images
- **Fear**: 4,097 images
- **Happy**: 7,215 images
- **Neutral**: 4,965 images
- **Sad**: 4,830 images
- **Surprise**: 3,171 images

#### Test Set
- **Angry**: 958 images
- **Disgust**: 111 images
- **Fear**: 1,024 images
- **Happy**: 1,774 images
- **Neutral**: 1,233 images
- **Sad**: 1,247 images
- **Surprise**: 831 images

## Image Dimensions
- **Unique Dimensions**: 1
- **Most Common**: (48, 48) (35887 images)
- **Width Range**: 48 - 48 (mean: 48.0)
- **Height Range**: 48 - 48 (mean: 48.0)

## File Size Analysis
- **Size Range**: 359 - 2,483 bytes
- **Mean Size**: 1,575 bytes
- **Median Size**: 1,578 bytes
- **Std Deviation**: 162 bytes

## Pixel Value Analysis
- **Value Range**: 0 - 255
- **Mean Min Value**: 7.73
- **Mean Max Value**: 236.74
- **Normalization Status**: Raw values (0-255)

## Recommendations
- ⚠️  **Pixel values are not normalized** (0-255 range detected)
- ✅ **Consistent image dimensions** across dataset
- ⚠️  **Class imbalance detected** (ratio: 16.5:1)
