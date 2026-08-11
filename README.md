# Practicing AI

### 📘 Overview
This project provides a hands-on introduction to three key areas of Artificial Intelligence: Traditional Machine Learning, Deep Learning, and Generative AI. Each section includes both theoretical insights and practical applications using Jupyter notebooks.

#### 1. 🧠 Traditional Machine Learning
Supervised vs. Unsupervised learning

Understanding the dataset: missing values, distributions

Data exploration and cleaning

Feature selection & feature engineering

Performance improvements: normalization, scaling, train/test split

Choosing baseline models (e.g., Random Forest)

Evaluation metrics and result analysis

🛠 Practice: A complete Jupyter notebook that walks through all the above steps with explanations.

#### 2. 🧬 Deep Learning – Image or Audio Classification with CNNs
Deep Learning vs. Traditional ML

What convolutional layers are and how they process pixels

From raw input to accurate classification

Visualizations and plots

Evaluation metrics and result analysis

Best practices: data augmentation, handling imbalanced datasets, common issues and fixes

🛠 Practice: A Jupyter notebook applying these steps on an image classification task with multiple classes.


#### 3. 🤖 Generative AI – Chatbots and Simple RAGs
Introduction to Generative AI concepts

Why language models hallucinate, and what Retrieval-Augmented Generation (RAG) fixes

Embeddings, chunking, vector stores and semantic retrieval

Building a RAG pipeline end to end: ingest → split → embed → retrieve → answer

Conversational RAG: handling follow-up questions and query rewriting

Evaluating a RAG system and analysing its failure modes

🛠 Practice: A guided notebook (`04-RAG/simple_RAG.ipynb`) that first teaches the concepts with runnable
examples, then provides five exercise sections to build, tune, extend and evaluate your own RAG system
over a local document collection. Runs fully offline with Ollama, or with a free-tier API if your machine
is limited. See [`04-RAG/README.md`](./04-RAG/README.md) for setup.
