# Training Progression Analysis Report
Generated: 2025-10-08 11:36:08

## Experiment Summary

### Experiment 1: demo_disgust_surprise_20251008_110604
- **Description**: Phase 1: Promoting disgust and surprise, demoting happy
- **Epochs**: 3
- **Resume from**: None
- **Emotion Focus**:
  - disgust: 3.0x (Promoted)
  - happy: 0.3x (Demoted)
  - surprise: 2.5x (Promoted)
- **Final Accuracy**: 0.4957
- **Best Accuracy**: 0.4957

### Experiment 2: demo_fear_focus_20251008_111115
- **Description**: Phase 2: Demoting disgust, promoting fear, continuing to demote happy
- **Epochs**: 5
- **Resume from**: ./experiments/demo_disgust_surprise_20251008_110604/checkpoint_epoch_3.pt
- **Emotion Focus**:
  - disgust: 0.3x (Demoted)
  - fear: 3.0x (Promoted)
  - happy: 0.5x (Demoted)
- **Final Accuracy**: 0.5386
- **Best Accuracy**: 0.5386

## Training Progression Analysis

### Key Observations:

- **Overall Improvement**: +0.0429 (+8.6%)
- **Training Success**: Model performance improved across experiments
