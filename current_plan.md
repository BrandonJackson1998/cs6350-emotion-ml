# Full Training Dataset Plan

## Change Log
- **2025-10-14**: Initial plan created for using all training images per epoch instead of limited 100 samples per class
- **2025-10-14**: Added enhanced checkpointing system for early stopping and mid-epoch recovery

## Overview
Modify the current emotion recognition training pipeline to utilize the complete training dataset (28,709 images) instead of the current limitation of 100 samples per class (700 images total). This will test the impact of using the full dataset on model performance and training dynamics.

## Goals
- **Primary**: Implement full dataset training mode that uses all 28,709 training images per epoch
- **Performance**: Compare training dynamics and final accuracy against current 100-sample baseline
- **Validation**: Ensure stable training with significantly larger dataset (40x increase in data volume)
- **Efficiency**: Maintain reasonable training time for single epoch test run
- **Reliability**: Enable safe early stopping and resume capabilities for long training runs

## Requirements

### Functional Requirements
- Remove `max_samples_per_class=100` limitation from training dataset
- Support both limited and full dataset modes via configuration flag
- Maintain existing checkpoint system and experiment tracking
- Preserve backward compatibility with existing experiment configurations
- **NEW**: Implement intra-epoch checkpointing for safe interruption and resume
- **NEW**: Add graceful handling of training interruptions (Ctrl+C, system shutdown)

### Non-Functional Requirements
- **Memory**: Handle increased batch loading without OOM errors
- **Performance**: Complete single epoch in reasonable time (<2 hours on available hardware)
- **Storage**: Accommodate larger checkpoint files and training history logs
- **Monitoring**: Enhanced progress tracking for longer training duration
- **Reliability**: Checkpoint system must handle mid-epoch saves and recovery

### Current Dataset Distribution
**Training Set (28,709 total images)**:
- Happy: 7,215 images (25.1%)
- Neutral: 4,965 images (17.3%) 
- Sad: 4,830 images (16.8%)
- Fear: 4,097 images (14.3%)
- Angry: 3,995 images (13.9%)
- Surprise: 3,171 images (11.0%)
- Disgust: 436 images (1.5%)

**Class Imbalance Ratio**: 16.5:1 (Happy vs Disgust)

## Risks & Mitigations

### High Risk
- **Memory Overflow**: 40x data increase may exceed GPU/RAM capacity
  - *Mitigation*: Monitor memory usage, reduce batch size if needed
- **Training Instability**: Large class imbalance (16.5:1 ratio) may cause poor convergence
  - *Mitigation*: Implement focal loss or class weighting alongside full dataset
- **Overfitting**: Extended exposure to full dataset in single epoch
  - *Mitigation*: Monitor validation metrics closely, early stopping if needed
- **NEW**: **Training Interruption**: Long training runs vulnerable to system crashes, power loss
  - *Mitigation*: Implement batch-level checkpointing every 100 batches (~10-15 minutes)

### Medium Risk  
- **Training Time**: Single epoch may take 1-2 hours vs current ~10 minutes
  - *Mitigation*: Run test during off-peak hours, implement progress checkpointing
- **Disk I/O Bottleneck**: 28K image loads per epoch vs current 700
  - *Mitigation*: Optimize DataLoader num_workers, consider SSD storage
- **NEW**: **Checkpoint Storage**: Frequent batch checkpoints may consume significant disk space
  - *Mitigation*: Implement checkpoint cleanup (keep only last 3 batch checkpoints)

### Low Risk
- **Experiment Tracking**: Larger log files and checkpoint sizes
  - *Mitigation*: Implement log rotation, checkpoint compression

## Implementation Steps

### Phase 1: Code Modification (2-3 hours)
- [ ] Add `use_full_dataset` parameter to `create_experiment_config()`
- [ ] Modify `FER2013FolderDataset.__init__()` to conditionally remove max_samples_per_class limitation
- [ ] Update `run_experiment()` to handle full dataset configuration
- [ ] Add progress estimation for longer training epochs
- [ ] Implement memory usage monitoring and reporting

#### **NEW**: Enhanced Checkpointing for Full Dataset Training
- [ ] Implement batch-level checkpoint saving (every 100 batches)
- [ ] Add batch progress tracking to checkpoint state (`current_batch`, `total_batches`)
- [ ] Create checkpoint cleanup strategy (keep last 3 batch checkpoints)
- [ ] Add signal handlers for SIGINT (Ctrl+C) and SIGTERM for graceful interruption
- [ ] Implement automatic checkpoint save on interruption
- [ ] Add training resume detection from partial epoch
- [ ] Modify `load_checkpoint()` to support mid-epoch resume
- [ ] Implement checkpoint compression for large states
- [ ] Add checkpoint validation to prevent corruption

### Phase 2: Configuration Setup (30 minutes)
- [ ] Create full dataset experiment configuration
- [ ] Set conservative batch size (16 vs current 32) to prevent OOM
- [ ] Configure single epoch test run (num_epochs=1)
- [ ] Enable detailed logging and checkpointing

#### **NEW**: Checkpoint Configuration
- [ ] Set `checkpoint_frequency = 100` batches (every ~10-15 minutes)
- [ ] Enable `auto_checkpoint_on_interrupt = True`
- [ ] Configure `max_batch_checkpoints = 3` for storage management
- [ ] Set `checkpoint_compression = True` for space efficiency

### Phase 3: Test Execution (1-2 hours)
- [ ] Run baseline experiment (100 samples/class) for comparison
- [ ] Execute full dataset single epoch experiment
- [ ] Monitor system resources (GPU memory, CPU, disk I/O)
- [ ] Capture training metrics and convergence behavior
- [ ] **NEW**: Test manual interruption and resume capability

### Phase 4: Analysis & Validation (30 minutes)
- [ ] Compare training loss convergence patterns
- [ ] Analyze batch-level accuracy progression
- [ ] Evaluate memory and performance characteristics
- [ ] Document findings and recommendations
- [ ] **NEW**: Validate checkpoint system reliability and resume accuracy

## Testing Strategy

### Unit Tests
- [ ] Test dataset loading with `use_full_dataset=True`
- [ ] Verify class count reporting shows full dataset sizes
- [ ] Confirm memory usage monitoring functions
- [ ] **NEW**: Test batch checkpoint save/load functionality
- [ ] **NEW**: Verify signal handler registration and cleanup
- [ ] **NEW**: Test checkpoint compression and validation

### Integration Tests  
- [ ] End-to-end training run with small subset (1000 images)
- [ ] Checkpoint save/load with larger dataset configuration
- [ ] Experiment tracking and logging with extended data
- [ ] **NEW**: Manual interruption test (Ctrl+C after 200 batches, then resume)
- [ ] **NEW**: Mid-epoch checkpoint recovery validation
- [ ] **NEW**: Checkpoint cleanup automation testing

### Performance Tests
- [ ] Memory profiling during full dataset loading
- [ ] Timing analysis for single epoch completion
- [ ] GPU utilization monitoring throughout training
- [ ] **NEW**: Checkpoint save/load performance impact measurement
- [ ] **NEW**: Storage space usage monitoring for batch checkpoints

### Test Data & Environments
- **Test Environment**: Current development setup with GPU acceleration
- **Test Dataset**: Full FER2013 training set (28,709 images)
- **Comparison Baseline**: Current 100-sample experiment results
- **Success Metrics**: Successful epoch completion, stable memory usage, measurable training progress
- **NEW**: **Reliability Metrics**: Successful interruption/resume cycles, checkpoint integrity validation

## Rollout & Monitoring

### Release Plan
1. **Local Testing**: Single full epoch run on development environment
2. **Validation**: Compare results with baseline 100-sample experiment  
3. **Documentation**: Update experiment runner with new capability
4. **Integration**: Merge full dataset support into main training pipeline

### Feature Flags
- `use_full_dataset`: Boolean flag to enable/disable full dataset mode
- `conservative_batch_size`: Reduced batch size for memory safety
- `enhanced_monitoring`: Extended logging and resource tracking
- **NEW**: `batch_checkpointing`: Enable/disable intra-epoch checkpointing
- **NEW**: `auto_resume`: Automatically detect and resume from last checkpoint

### Metrics & Alerts
- **Training Metrics**: Loss convergence, accuracy progression, epoch timing
- **System Metrics**: GPU memory usage, CPU utilization, disk I/O rates
- **Alert Thresholds**: >90% GPU memory usage, training time >3 hours
- **NEW**: **Checkpoint Metrics**: Checkpoint save time, storage usage, validation success rate

### Rollback Steps
1. Terminate training run if memory issues occur
2. Revert to 100-sample configuration for stable baseline
3. Investigate resource bottlenecks before retry
4. Scale batch size down incrementally if needed

#### **NEW**: Emergency Procedures
1. **Manual Stop**: Ctrl+C triggers automatic checkpoint save before termination
2. **Resume Training**: Restart script automatically detects and loads last batch checkpoint
3. **Checkpoint Recovery**: Validate and repair corrupted checkpoints using backup copies
4. **Storage Management**: Automated cleanup of intermediate batch checkpoints
5. **Rollback**: Disable batch checkpointing if storage/performance issues occur

## Open Questions

### Technical Decisions Needed
- **Batch Size Optimization**: What's the optimal batch size for 28K images? (Propose: Start with 16, test 8 if OOM)
- **DataLoader Workers**: How many num_workers for optimal I/O performance? (Propose: Test 2, 4, 8 workers)
- **Memory Management**: Should we implement gradient accumulation for smaller batch sizes? (Propose: Not initially, keep simple)
- **NEW**: **Checkpoint Frequency**: Is 100 batches optimal, or should we adjust based on batch size? (Propose: Target 10-15 minute intervals)
- **NEW**: **Resume Strategy**: Should resume restart from exact batch or beginning of checkpoint interval? (Propose: Exact batch for precision)

### Next Actions Required
1. **Resource Assessment**: Confirm available GPU memory and storage capacity
2. **Baseline Documentation**: Capture current 100-sample experiment metrics for comparison
3. **Monitoring Setup**: Implement system resource tracking before full dataset run
4. **Timeline Coordination**: Schedule 2-3 hour window for full dataset test execution
5. **NEW**: **Checkpoint Testing**: Validate interrupt/resume cycle with small dataset first
6. **NEW**: **Storage Planning**: Estimate checkpoint storage requirements and cleanup policies

### Dependencies
- Current checkpoint system must handle larger model states
- Experiment tracking system should accommodate longer training logs  
- Available compute resources must support 40x data volume increase
- **NEW**: Signal handling libraries (already available in Python standard library)
- **NEW**: Sufficient disk space for multiple batch checkpoints (estimate: 500MB per checkpoint × 3 = 1.5GB)

**Estimated Total Timeline**: 5-6 hours (2-3 hrs implementation + 1-2 hrs testing + 1 hr analysis)

**Success Criteria**: Successfully complete single epoch training on full dataset with stable memory usage, meaningful training progress compared to 100-sample baseline, and validated interrupt/resume capability.