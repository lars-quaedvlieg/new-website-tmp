---
visible: true
layout: page
title: "On the Effect of Quantization on Deep Leakage from Gradients and Generalization"
authors: Lars C.P.M. Quaedvlieg, Arvind Menon
description: We explore various quantization techniques and assess their effectiveness in preserving both data privacy and model performance for machine learning.
img: assets/img/optml_project.png
importance: 95
category: Research
github: https://github.com/arvind6599/Opt_ML_Project/
developed_date: 2024-07-17 16:00:00-0000
---

This project was done for the CS-439 Modern Natural Language Processing course at EPFL in Spring 2024.

## Key Contributions

- **Quantization Techniques**: We introduce and evaluate several quantization strategies including Uniform Quantization and Stochastic Rounding, assessing their impact on the privacy-security trade-off in neural network training.
- **Model Performance**: Our findings indicate that quantization can maintain model accuracy compared to other defense mechanisms, offering a promising solution to mitigate privacy risks without significant performance drawbacks.
- **In-depth Analysis**: We provide a comprehensive analysis of the Deep Leakage from Gradients (DLG) threat model, including scenarios where traditional defenses either fail or lead to degraded model performance.

## Methodology

1. **Gradient Quantization**: We apply different levels of gradient quantization to understand how they affect the ability to reconstruct training data from shared gradients.
2. **Comparative Analysis**: The effectiveness of each quantization method is compared against baseline models and those subjected to sparsity and noise addition.
3. **Experimental Setup**: Utilizing CIFAR-10 and synthetic datasets, we evaluate under real-world conditions to ensure robustness and applicability of our conclusions.

## Results

- **Enhanced Privacy**: Quantization significantly reduces the risk of sensitive data leakage, similarly to other methods, without adversely affecting the training process.
- **Performance Metrics**: Our quantized models perform comparably to non-quantized trained models in terms of accuracy and training stability.

Please see the final project report below for more in-depth information:

{% pdf "/assets/pdf/optml_project.pdf" height=1030px no_link %}