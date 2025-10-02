# cs6350-emotion-ml
Detecting change in emotion given a time series of images

## First Run

Classification Report:
              precision    recall  f1-score   support

       angry     0.4390    0.5400    0.4843       100
     disgust     0.8972    0.9600    0.9275       100
        fear     0.3934    0.2400    0.2981       100
       happy     0.8370    0.7700    0.8021       100
     neutral     0.5088    0.5800    0.5421       100
         sad     0.3981    0.4100    0.4039       100
    surprise     0.7800    0.7800    0.7800       100

    accuracy                         0.6114       700
   macro avg     0.6076    0.6114    0.6054       700
weighted avg     0.6076    0.6114    0.6054       700


Classification report saved to './outputs/classification_report.txt'
Confusion matrix saved as './outputs/confusion_matrix_benchmark.png'
Training history saved as './outputs/training_history_benchmark.png'
Training history data saved to './outputs/training_history.csv'

==================================================
Benchmark complete!
All outputs saved to: ./outputs
==================================================



## Project TODO List

### 1. Problem Definition & Goals
- [ ] Define emotion transition detection for educational monitoring
  - Detect transitions between 7 emotional states: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - Focus on monitoring students' emotional states during learning sessions
- [ ] Establish success criteria and evaluation metrics
  - Primary metric: Accuracy of emotion transition detection
  - Secondary metric: Processing speed for real-time application
- [ ] Define target emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- [ ] Set performance benchmarks
  - Target accuracy: >85% for emotion classification
  - Target speed: Real-time processing capability (<100ms per frame)
  - Application: Educational technology for student engagement monitoring

### 2. Data Collection
- [ ] Download FER-2013 dataset from Kaggle
  - Source: https://www.kaggle.com/datasets/msambare/fer2013/data?select=test
  - Target: ~30,000 images for training, ~10,000 for testing
- [ ] Verify dataset contains all required emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- [ ] Document dataset characteristics and limitations
- [ ] Set up data storage structure and organization
- [ ] Validate data integrity and completeness

### 3. Data Preparation
- [ ] Clean and validate image data
- [ ] Preprocess images (resize, normalize, etc.)
- [ ] Handle missing or corrupted data
- [ ] Create train/validation/test splits
- [ ] Format data for model consumption

### 4. Feature Engineering
- [ ] Extract facial features from images
- [ ] Create temporal features from time series
- [ ] Select most informative features
- [ ] Handle feature scaling and normalization
- [ ] Create feature pipelines

### 5. Model Training
- [ ] Select appropriate ML algorithms
- [ ] Implement baseline models
- [ ] Train deep learning models (CNNs, RNNs, etc.)
- [ ] Handle time series aspects of the data
- [ ] Cross-validation setup

### 6. Evaluation
- [ ] Implement evaluation metrics
- [ ] Create test harness
- [ ] Perform model validation
- [ ] Compare model performances
- [ ] Generate evaluation reports

### 7. Hyperparameter Tuning
- [ ] Define hyperparameter search space
- [ ] Implement tuning strategy (grid search, random search, etc.)
- [ ] Optimize model parameters
- [ ] Document best parameters found

### 8. Deployment
- [ ] Package model for deployment
- [ ] Create inference pipeline
- [ ] Build API or application interface
- [ ] Test deployment setup
- [ ] Create documentation for usage

### 9. Monitoring & Maintenance
- [ ] Implement performance monitoring
- [ ] Set up model drift detection
- [ ] Create update/retraining pipeline
- [ ] Plan for model versioning
- [ ] Document maintenance procedures
