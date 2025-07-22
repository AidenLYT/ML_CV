# Classification with ResNet34

This project implements a learning pipeline using a **modified ResNet34 model** for image classification across **50 classes**. It leverages transfer learning, advanced data augmentation, and a custom classification head for robust performance on small datasets.

## Model: `R34_ver1`

The model uses the ResNet34 backbone (pretrained on ImageNet) and appends a custom classifier.

### Features:
1. **Backbone freezing**: Option to freeze feature extractor for early epochs.
2. **Adaptive Average Pooling**: Converts feature maps to fixed-length vectors.
3. **Custom Classification Head**:
    - Dropout
    - BatchNorm
    - ReLU activations
    - Fully-connected layers

### Architecture Flow:
[Input Image] → ResNet34 Backbone → AvgPool → Custom Classifier → [Logits]

