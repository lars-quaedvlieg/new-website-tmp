---
visible: true
layout: page
title: "Chess ratings: Leveraging Network Methods to Predict Chess Results"
authors: Lars C.P.M. Quaedvlieg, Lazar Milikic
description: We address the problem of estimating chess player ratings and match outcome prediction by leveraging network approaches based on their past games and results.
img: assets/img/network-ml.png
importance: 97
category: Research
developed_date: 2023-07-01 16:00:00-0000
---

This project was done for the EE-452 Network machine learning course at EPFL in Fall 2023.

In our final project for EE-452, we focused on **estimating chess player ratings and predicting match outcomes by applying 
network methods to historical game data**. We proposed a rank regression framework that not only *predicts match outcomes* 
based on input features but also *learns the player ratings implicitly*. Our approach included the use of hand-crafted 
features and graph neural networks (GNNs) to learn embeddings, which were then used to train the regression model.

We evaluated the performance of our models on a dataset consisting of 65,053 games played over 100 months among 
approximately 7,000 players, structured into a network graph where each player is a node and each game is a directed 
edge with attributes indicating the game's outcome and timing. Our analyses revealed several network properties, 
including a *heavy-tailed degree distribution* and a *scale-free nature*, suggesting non-random connections and *strong 
community structures* within the network.

We compared our network with standard network models like Erdős-Rényi, Watts-Strogatz, and Barabási-Albert, finding 
that none perfectly matched the characteristics of our chess games network. We also experimented with various node properties
and community detection methods to understand their predictive power on match outcomes and player ratings.

Our experiments included developing a baseline model using hand-crafted features and advancing to more sophisticated
models using **unsupervised node embeddings** and an **end-to-end GNN approach**. We incorporated various regularization techniques
to prevent overfitting and to account for the temporal dynamics of player performance.

The success of this project underscores the potential of network-based approaches in predictive analytics for chess and similar
domains where historical performance data can be structured as a network. We conclude that leveraging network properties 
through carefully designed features and advanced modeling techniques can provide significant insights and predictive power
in rating systems.

Please see the report below for more information:

{% pdf "/assets/pdf/network_ml_project.pdf" height=1030px no_link %}