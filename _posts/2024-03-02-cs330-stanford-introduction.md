---
layout: distill
title: "CS-330: Deep Multi-Task and Meta Learning - Introduction"
description: I have been incredibly interested in the recent wave of multimodal foundation models, especially in robotics and sequential decision-making. Since I never had a formal introduction to this topic, I decided to audit the Deep Multi-Task and Meta Learning course, which is taught yearly by Chelsea Finn at Stanford. I will mainly document my takes on the lectures, hopefully making it a nice read for people who would like to learn more about this topic!
date: 2024-03-01
tags: course
categories: deep-multi-task-and-meta-learning
comments: true

authors:
  - name: Lars C.P.M. Quaedvlieg
    url: "https://lars-quaedvlieg.github.io/"
    affiliations:
      name: EPFL
      
bibliography: blog/cs330/2024-03-01-introduction.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Lectures
  - name: Why multi-task and meta-learning?
  - name: What are tasks?
  - name: References

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

## Introduction

The course [CS 330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/), by [Chelsea Finn](https://ai.stanford.edu/~cbfinn/), is taught
on a yearly basis and discusses the foundations and current state of multi-task learning and meta learning.

**:warning: Note:** I am discussing the content of the edition in Fall 2023, which no longer includes reinforcement learning.
If you are interested in this, I will be auditing [CS 224R Deep Reinforcement Learning](https://cs224r.stanford.edu/)
later this spring, which is also taught by [Chelsea Finn](https://ai.stanford.edu/~cbfinn/).

In an attempt to improve my writing skills and provide useful summaries/voice my opinions, I have decided to discuss 
the content of every lecture in this blog. In this post, I will give an overview of the course and why it is important 
for AI, especially now.

This course will focus on solving problems that are composed of multiple tasks, and studies how structure that arises from these multiple tasks can be leveraged to learn more efficiently/effectively, including:

- Self-supervised pre-training for downstream few-shot learning and transfer learning.
- Meta-learning methods that aim to learn efficient learning algorithms that can learn new tasks quickly.
- Curriculum and lifelong learning, where the problem requires learning a sequence of tasks, leveraging their shared structure to enable knowledge transfer.

***

## Lectures

The lecture schedule of the course is as follows:
1. [Multi-task learning](/blog/2024/cs330-stanford-mtl/)
2. [Transfer learning & meta learning](/blog/2024/cs330-stanford-tl-ml/)
3. [Black-box meta-learning & in-context learning](/blog/2024/cs330-stanford-bbml-icl/)
4. [Optimization-based meta-learning](/blog/2024/cs330-stanford-obml/)
5. [Few-shot learning via metric learning](/blog/2024/cs330-stanford-fsl-ml/)
6. [Unsupervised pre-training for few-shot learning (contrastive)](/blog/2024/cs330-stanford-upt-fsl-cl/) 
7. [Unsupervised pre-training for few-shot learning (generative)](/blog/2024/cs330-stanford-upt-rbm/)
8. Advanced meta-learning topics (task construction)
9. Variational inference
10. Bayesian meta-learning
11. Advanced meta-learning topics (large-scale meta-optimization)
12. Lifelong learning
13. Domain Adaptation and Domain Generalization
14. Frontiers & Open Challenges

I am excited to start discussing these topics in greater detail! Check this page regularly for updates, since I will 
link to new posts whenever they are available!

***

## Why multi-task and meta-learning?

{% include figure.liquid path="assets/img/blog/cs330/1/robotics_example.png" zoomable=true %}

Robots are embodied in the real world, and must generalize across tasks. In order to do so, they need some common sense 
understanding and supervision can’t be taken for granted.

Earlier robotics and reinforcement research mainly focused on problems that required learning a task from scratch. This 
problem is even present in other fields, such as object detection or speech recognition. However, as opposed to these 
problems, **humans are generalists** that exploit common structures to solve new problems more efficiently.

Going beyond the case of generalist agents, deep multi-task and meta learning useful for any problems where a **common 
structure** can benefit the efficiency or effectiveness of a model. It can be impractical to develop models for each
specific task (e.g. each robot, person, or disease), especially if the data that you have access to for these individual
tasks is **scarce**.

If you need to **quickly learn something new**, you need to utilize prior experiences (e.g. few-shot learning) to make 
decisions.

But why now? Right now, with the speed of research advancements in AI, many researchers are looking into utilizing 
multi-model information to develop their models. Especially in robotics, foundation models seem **the** topic in 2024,
and many advancements have been made in the past year <d-cite key="zhao2023learning"></d-cite>, <d-cite key="open_x_embodiment_rt_x_2023"></d-cite>, <d-cite key="octo_2023"></d-cite>, <d-cite key="brohan2023rt"></d-cite>.

***

## What are tasks?

Given a dataset $\mathcal{D}$ and loss function $\mathcal{L}$, we hope to develop a model $f_\theta$. Different tasks 
can be used to train this model, with some simple examples being objects, people, objectives, lighting conditions, 
words, languages, etc.

The **critical assumption** here is that different tasks must share some common structure. However, in practice, this 
is very often the case, even for tasks that seem unrelated. For example the laws of physics and the rules of English
can be shared among many tasks.

1. The multi-task problem: Learn **a set of tasks** more quickly or more proficiently than learning them independently.
2. Given data on previous task(s), learn **a new task** more quickly and/or more proficiently.

> Doesn’t multi-task learning reduce to single-task learning?

This is indeed the case when aggregating data across multiple tasks, which is actually one approach to multi-task 
learning. However, what if you want to learn new tasks? And how do you tell the model which task to do? And what if 
aggregating doesn’t work?

***