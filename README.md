# Comparative Study of Markov Models and Transformers for Long-Range Sequence Modeling

## 📌 Overview

This project presents a controlled empirical comparison between classical **first-order Markov models** and modern **Transformer architectures** for sequence prediction tasks.

The goal is to demonstrate the **limitations of fixed-memory probabilistic models** and show how **self-attention with positional encoding** enables Transformers to model **long-range dependencies** that Markov models fundamentally cannot.

---

## 🎯 Problem Statement

Given a sequence of binary tokens, predict the next token.

We compare the performance of:
- A **first-order Markov model**
- A **Transformer with self-attention**

on the same dataset, specifically designed to require **long-range dependency modeling**.

---

## 🧠 Key Idea

Markov models assume a fixed-order dependency:

`P(x_t | x_{t-1})`


Transformers learn dependencies over the entire history:

`P(x_t | x_1, x_2, ..., x_{t-1})`

When the data violates the Markov assumption, Markov models fail, while Transformers succeed by learning **adaptive memory through attention**.

---

## 🧪 Dataset Design

The dataset is synthetically generated to explicitly violate the Markov property.

The next token is defined as:

`x_t = x_{t-3} ⊕ x_{t-10}`


Where:
- ⊕ denotes XOR
- The previous token alone gives no information
- Only distant past tokens determine the next value

This ensures that **short-memory models cannot solve the task**.

---

## 🏗️ Models Implemented

### 🔹 First-Order Markov Model
- Transition probability matrix
- Stationary distribution assumption
- Fixed memory (one-step dependency)

### 🔹 Transformer
- Token embeddings
- Sinusoidal positional encoding
- Masked self-attention
- Stacked Transformer blocks
- Trained as a next-token predictor

All Transformer components are implemented **from scratch** (without using `nn.Transformer`).

---

## 📊 Results

| Model | Accuracy |
|------|----------|
| First-order Markov Model | ~0.50 |
| Transformer | **1.00** |

### Interpretation
- The Markov model performs at chance level due to violated assumptions
- The Transformer perfectly learns the long-range XOR dependency

---

## 🔍 Training Behavior

- Early epochs show high loss (random guessing)
- Sudden loss collapse indicates discovery of the underlying structure
- Final loss approaching zero confirms exact rule learning

This behavior is characteristic of learning **compositional long-range dependencies**.

---

## 🧠 Key Insights

- Self-attention without positional encoding is insufficient
- Positional encoding is essential for temporal structure
- Model depth enables composition of non-linear dependencies
- Transformers act as adaptive memory systems
- Model assumptions critically affect performance

---

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Experiment
```bash
python3 -m experiments.compare_models
```
