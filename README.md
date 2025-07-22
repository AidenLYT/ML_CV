This project implements a learning pipeline using a modified ResNet34 model for image classification across 50 classes.

#Model: R34
1. Backbone freezing
2. Adaptive average pooling
3. Custom classification head
   3.1 Dropout
   3.2 BatchNorm
   3.3 ReLU activations
   
[Input Image] → ResNet34 Backbone → AvgPool → Custom Classifier → [Logits]
