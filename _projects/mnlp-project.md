---
visible: true
layout: page
title: "ProbLLMs: Multiple-Choice Problem Solving for EPFL Courses"
authors: Lars C.P.M. Quaedvlieg, Lazar Milikic, Somesh Mehra, Arvind Menon
description: We develop an AI tutor targeted at STEM education, specifically for multiple-choice question answering related to EPFL courses. Using a general- purpose LLM as a base, we fine-tune a model with enhanced capabilities for complex reasoning tasks related to STEM education.
img: assets/img/mnlp.png
importance: 95
category: Research
github: https://github.com/lars-quaedvlieg/ProbLLMs
developed_date: 2024-07-17 16:00:00-0000
---

This project was done for the CS-552 Modern Natural Language Processing course at EPFL in Spring 2024.

## Innovation and Development

### 1. Fine-Tuning on Specialized STEM Data

We started with a general-purpose Large Language Model (LLM) and enhanced it for STEM education by fine-tuning it on 
a curated dataset including:
- Complex problem-solving questions.
- Theory and application-based questions from EPFL course materials.
- External datasets to broaden the model's problem-solving capabilities.

### 2. Retrieval Augmented Generation (RAG)

To enhance the AI tutor's accuracy and relevance, we integrated the RAG technique which dynamically pulls in pertinent 
information from a knowledge base while the AI generates answers. This method ensures:
- Higher accuracy by using up-to-date and course-specific information.
- Detailed explanations that help students understand the reasoning behind each answer.

### 3. Model Quantization

Understanding the need for accessibility, we implemented quantization techniques to reduce the model’s resource 
requirements, allowing it to perform well even in lower-resource environments. This step is crucial for facilitating:
- Easier deployment of the AI tutor across various platforms.
- Use in regions with limited access to high computational power.

## Achievements and Impact

- **Performance:** Our enhanced model outstripped the base LLM in complex reasoning tasks, proving its efficacy in a rigorous academic setting.
- **Accessibility:** By reducing computational demands through quantization, our AI tutor can be used on less powerful devices, making STEM education more accessible.
- **Educational Enhancement:** The AI tutor provides students with immediate, high-quality academic support, especially beneficial in large classes or where educational resources are limited.

Please see the reports below for more information. We provide three different reports. The first presents the final (full project) report,
the second is a progress report, and the third is the initial project proposal.

### Final Paper

{% pdf "/assets/pdf/mnlp_project.pdf" height=1030px no_link %}

### Progress Report

{% pdf "/assets/pdf/mnlp_progress.pdf" height=1030px no_link %}

### Project Proposal

{% pdf "/assets/pdf/mnlp_proposal.pdf" height=1030px no_link %}