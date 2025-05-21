---
visible: true
layout: page
title: Discovering the Higgs Boson using Machine Learning
authors: Lars. C.P.M. Quaedvlieg, Anton Pirhonen, Ana S. Leiva
description: This project studies different machine learning models applied to the data collected from the experiments performed with the CERN particle accelerator with the aim of discovering the Higgs boson particle
img: assets/img/cern-lhc.jpg
importance: 99
category: Research
developed_date: 2022-10-31 16:00:00-0000
repository_name: CS-433/ml-project-1-cheruben-3
github: https://github.com/CS-433/ml-project-1-cheruben-3
---

This project was completed for the [CS-433 Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) course 
taught at EPFL in Fall 2022. The goal of this project is to learn to use basic machine learning concepts on a
real-world dataset, start to finish. First, an **exploratory data analysis** is performed to understand your dataset and the
features. Then, **feature processing and engineering** is done to clean the dataset and extract more meaningful information.
Afterwards, linear and logistic regression are **implemented** with some tweaks and used on the data. Finally, the methods 
are **analyzed and inference** is performed.

The final report for this project is embedded below. It focuses on the most important points but does not exhaustively 
cover everything done in the project. For more information on the project implementations, visit the GitHub repository 
linked above.

Here are some keypoint of the project:

- Higgs Boson Classification
- Ridge Regression
- Ridge Logistic Regression
- Kernels
- Accelerated Gradient Descent with Restart

The main finding from this project is that basic methods like linear or logistic regression can still be used to create 
a powerful non-linear model for binary classification using kernel methods. This simple approach can compete with more 
sophisticated models like neural networks and random forests while being much simpler to implement and understand.

{% pdf "/assets/pdf/ml_epfl_project_1.pdf" height=1030px no_link %}