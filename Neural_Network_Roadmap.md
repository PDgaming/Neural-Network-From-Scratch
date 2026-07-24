# Neural Network Learning Roadmap

# Phase 1 -- Finish the MLP

Before moving to CNNs, turn your current MLP into a solid training framework.

## Optimization

-   Momentum
-   RMSProp
-   Adam
-   Learning-rate scheduling

## Initialization

-   Xavier Initialization
-   He Initialization

---

# Phase 2 -- Practice Datasets

Recommended progression:

1.  Your childhood dataset
2.  Iris
3.  Wine
4.  Breast Cancer Wisconsin
5.  MNIST (using the MLP)

Goal: reach roughly 97--98% accuracy on MNIST before moving to CNNs.

---

# Phase 3 -- Convolutional Neural Networks

Learn:

-   Convolution
-   Filters/Kernels
-   Feature Maps
-   Padding
-   Stride
-   Pooling

Then solve MNIST again and compare the results with your MLP.

---

# Phase 4 -- Transformers

Learn:

-   Embeddings
-   Positional Encoding
-   Self-Attention
-   Multi-Head Attention
-   Feed-Forward Networks
-   Residual Connections
-   Layer Normalization

---

# Phase 6 -- Tiny LLM

Build progressively:

1.  Character-level language model
2.  Shakespeare text generator
3.  Small story generator
4.  Tiny code completion model

---


Even with manual backpropagation, structuring your project this way will make it much easier to add CNNs and, eventually, transformer layers.

---

# Priority Checklist

1.  Mini-batch training
2.  Softmax
3.  Cross-Entropy loss
4.  Adam optimizer
5.  He/Xavier initialization
6.  Validation and evaluation
7.  MNIST using the MLP
8.  CNNs
9.  Transformers
10. Tiny LLM
