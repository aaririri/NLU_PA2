# B22AI001-A2

## Overview

This repository contains implementations and experiments for word embedding models, including both library-based and from-scratch approaches. The focus is on understanding and comparing **CBOW** and **Skip-gram** architectures using a custom text corpus.

---

## Repository Structure

```
B22AI001-A2/
│── prob1.py
│── prob2.py
│── corpus.txt
│── TrainingNames.txt
│── README.md
```

### Files Description

#### 1. `prob1.py`

* Implements:

  * Corpus loading and inspection
  * Dataset statistics
  * Word cloud generation
* Acts as the **data preprocessing and exploration module**
* Helps verify correctness of corpus before training

---

#### 2. `prob2.py`

* Core training and evaluation pipeline
* Includes:

  * **Gensim-based models**

    * CBOW
    * Skip-gram
  * **From-scratch implementations**

    * Skip-gram with negative sampling
    * CBOW with negative sampling
* Supports:

  * Hyperparameter tuning (embedding size, window size, negative samples)
  * Nearest neighbor queries
  * Analogy tasks
  * Embedding visualization (PCA / t-SNE)

---

#### 3. `corpus.txt`

* Preprocessed text corpus
* Format:

  * One document per line
  * Tokens separated by spaces
* Example:

  ```
  this is a sample sentence
  another example text here
  ```

---

#### 4. `TrainingNames.txt`

* Contains:

  * Names / identifiers used during training or evaluation
* May be used for:

  * Filtering vocabulary
  * Testing similarity or analogy queries

---

## How to Run

### Step 1: Install Dependencies

```bash
pip install numpy matplotlib scikit-learn gensim wordcloud pymupdf torch
```

---

### Step 2: Run Data Inspection

```bash
python prob1.py
```

This will:

* Load the corpus
* Print sample documents
* Show frequent words
* Generate dataset statistics
* Display a word cloud

---

### Step 3: Train Models

```bash
python prob2.py
```

This will:

* Train multiple Word2Vec configurations
* Train scratch implementations
* Evaluate:

  * Nearest neighbors
  * Analogies
* Visualize embeddings

---

## Models Implemented

### 1. CBOW (Continuous Bag of Words)

* Predicts target word from context
* Faster training
* Works well for frequent words

### 2. Skip-gram

* Predicts context from target word
* Better for rare words
* Slower but more expressive

### 3. Scratch Implementations

* Built using PyTorch
* Includes:

  * Embedding layers
  * Negative sampling loss
* Useful for understanding internal mechanics

---

## Key Features

* Custom vocabulary construction with `min_count`
* Sliding window context generation
* Negative sampling for efficient training
* Cosine similarity for nearest neighbors
* Dimensionality reduction:

  * PCA
  * t-SNE

---

## Example Outputs

* Top similar words:

  ```
  research → study, analysis, work, project
  ```
* Analogy:

  ```
  undergraduate : btech :: postgraduate : mtech
  ```

---

## Notes

* Ensure `corpus.txt` is clean and tokenized before running
* Training scratch models can be computationally expensive
* Large hyperparameter combinations may increase runtime significantly

---

## Author

**B22AI001**

---
