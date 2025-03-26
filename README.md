```markdown
# Action Recognition with CNN-LSTM

A deep learning model for human action recognition using a hybrid CNN-LSTM architecture on the Weizmann dataset.

## Overview

This project implements a video action classification system that combines convolutional neural networks (CNN) for spatial feature extraction with long short-term memory networks (LSTM) for temporal sequence modeling. The model is trained on the Weizmann dataset, which contains 10 different human action classes.

## Action Classes
- walk
- run
- jump
- side
- bend
- wave1
- wave2
- pjump
- jack
- skip

## Model Architecture

The model uses a hybrid CNN-LSTM architecture:
1. **Feature Extraction**: Pretrained ResNet18 backbone
2. **Dimension Reduction**: 512→256 features
3. **Temporal Modeling**: 3-layer LSTM
4. **Classification Head**: Two fully-connected layers

## Implementation Details

### Data Processing
- Video splitting with strategic temporal sampling
- Temporal-consistent data augmentation
- Adaptive sequence length handling

### Training Methodology
- 5-fold stratified cross-validation
- Learning rate: 0.0002
- Optimizer: AdamW with weight decay
- Early stopping with patience of 10 epochs
- Learning rate scheduling with ReduceLROnPlateau

## Performance

- Best validation accuracy: 0.7733
- Average validation accuracy: 0.6411
- Test accuracy: 0.6774

## Requirements

```python
torch
torchvision
numpy
sklearn
matplotlib
seaborn
av
requests
tqdm
```

## Usage

1. **Data Preparation**:
```python
train_val_dataset, test_dataset = prepare_datasets(test_size=0.2, random_state=42)
```

2. **Training**:
```python
fold_train_losses, fold_train_accuracies, fold_val_losses, fold_val_accuracies, data_loaders = train_with_cross_validation(train_model=True)
```

3. **Inference**:
```python
predicted_class = classify(input_tensor, model_path='best_model.pth')
print(f"Predicted Class: {classes[predicted_class]}")
```

## Model Evaluation

```python
test_model(model_path='best_model.pth', test_loader=test_loader)
```

## Key Features

- Temporal-consistent data augmentation
- Strategic video sampling
- Transfer learning with ResNet18
- Robust cross-validation
- Comprehensive performance metrics

## Model Weights

Pre-trained model weights are available at:
`https://github.com/winstongarth/dl_coursework_2025_model/raw/refs/heads/main/pretrained_model.pth`
