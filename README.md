# 🧠 Sentiment Analysis for Mental Health

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-LSTM-D00000?logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-NLP-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project applies **Natural Language Processing (NLP)** and **machine learning** to classify mental health-related text statements into seven distinct sentiment categories: **Normal, Depression, Suicidal, Anxiety, Bipolar, Stress,** and **Personality Disorder**. The project is structured across three notebooks — EDA, Logistic Regression modeling, and RNN (LSTM) modeling — covering the complete NLP pipeline from raw text cleaning through model evaluation and post-analysis.

Two models are developed and compared: a classical **Logistic Regression** approach using TF-IDF vectorization, and a deep learning **LSTM** model using custom Word2Vec embeddings.

---

## 💼 Business Problem

Mental health conditions are among the most underdiagnosed and undertreated health issues globally. Individuals often express signs of depression, anxiety, suicidal ideation, and other conditions through written text on social platforms, forums, and messaging apps — long before they seek clinical help. Manual review of this content at scale is neither feasible nor timely.

There is a clear need for an automated, data-driven system that can detect mental health signals from text, enabling faster triage, proactive outreach, and better allocation of clinical resources.

---

## 🎯 Business Objective

Develop a multi-class NLP classification system that:

- Automatically classifies text statements into one of seven mental health sentiment categories
- Handles severe class imbalance without degrading performance on minority classes
- Compares classical (TF-IDF + Logistic Regression) and deep learning (LSTM + Word2Vec) approaches to identify the most effective method for mental health text classification
- Supports downstream use cases in digital mental health platforms, crisis detection tools, and clinical decision support systems

---

## 📦 Dataset

### Source

| Detail | Info |
|---|---|
| **File** | `sentiment_health.csv` |
| **Domain** | Mental health text classification |
| **Task** | Multi-class NLP sentiment classification |

### Dataset Summary

| Attribute | Details |
|---|---|
| Raw Records | ~53,000+ rows (before cleaning) |
| Cleaned Records | ~52,681 rows (after removing 362 missing statements) |
| Features | 2 columns: `statement` (text), `status` (label) |
| Target Classes | 7 sentiment categories |
| Train / Val / Test Split | 70% / 15% / 15% (stratified) |

**Target Class Distribution:**

| Class | Approx. Count | Imbalance Note |
|---|---|---|
| Normal | > 10,000 | Dominant class |
| Depression | > 10,000 | Dominant class |
| Suicidal | > 10,000 | Dominant class |
| Anxiety | ~3,000–4,000 | Moderate minority |
| Bipolar | ~3,000–4,000 | Moderate minority |
| Stress | ~1,000–2,000 | Minority class |
| Personality Disorder | ~1,000–2,000 | Minority class |

**Label Encoding:**

| Status | Encoded Label |
|---|---|
| Anxiety | 0 |
| Bipolar | 1 |
| Depression | 2 |
| Normal | 3 |
| Personality Disorder | 4 |
| Stress | 5 |
| Suicidal | 6 |

### Dataset Challenges

- **Missing Values:** 362 rows had null values in the `statement` column and were dropped before modeling. No missing values remained post-cleaning.
- **Severe Class Imbalance:** The dataset is dominated by Normal, Depression, and Suicidal classes (each exceeding 10,000 records), while Stress and Personality Disorder are significantly underrepresented. Left unaddressed, this biases models toward majority classes.
- **Minority Class Detection:** Classes like Anxiety, Stress, and Personality Disorder are harder to classify correctly due to their lower representation and linguistic overlap with dominant classes (e.g., Depression and Normal).
- **Unnamed Index Column:** The dataset included an unnamed index column requiring explicit renaming to `unique_id` during data exploration.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Text Preprocessing** | NLTK (stopwords), `re` (regex) |
| **Classical ML** | Scikit-Learn (TF-IDF, Logistic Regression, LabelEncoder, classification metrics) |
| **Deep Learning** | TensorFlow, Keras (LSTM, Embedding, Dropout, EarlyStopping, ReduceLROnPlateau) |
| **Word Embeddings** | Gensim (custom Word2Vec) |
| **Environment** | Jupyter Notebook |

---

## 🔄 Project Workflow

```
Notebook 1 — EDA
      ↓
Raw Data Loading
      ↓
Data Exploration (shape, dtypes, nulls)
      ↓
Missing Value Handling (drop 362 null statements)
      ↓
Class Distribution Analysis & Visualization
      ↓
Export Cleaned CSV

Notebook 2 — Logistic Regression
      ↓
Text Preprocessing (lowercase, stopwords, URLs, special chars, whitespace)
      ↓
Label Encoding (7 classes → integers)
      ↓
Stratified Train / Val / Test Split (70 / 15 / 15)
      ↓
TF-IDF Vectorization (max 30,000 features, unigrams + bigrams + trigrams)
      ↓
Logistic Regression with Hyperparameter Grid Search (C, solver)
      ↓
Evaluation (accuracy, classification report, confusion matrix)
      ↓
Post-Analysis (correct vs incorrect predictions)

Notebook 3 — RNN (LSTM)
      ↓
Text Preprocessing (same pipeline as Logistic Regression)
      ↓
Label Encoding + Stratified Split
      ↓
Tokenization (top 20,000 words, OOV token)
      ↓
Sequence Padding (max_tokens = mean + 2 × std ≈ 208, covers 96% of data)
      ↓
Custom Word2Vec Embeddings (32-dim, window=5, skip-gram)
      ↓
LSTM Architecture (Embedding → SpatialDropout → LSTM(64) → Dropout → Dense(7, softmax))
      ↓
Training with Class Weights + EarlyStopping + ReduceLROnPlateau
      ↓
Evaluation on Test Set (accuracy, confusion matrix)
      ↓
Post-Analysis (correct vs incorrect predictions)
```

---

## 📊 Exploratory Data Analysis

**Class Distribution**
The dataset is heavily imbalanced. Normal, Depression, and Suicidal each exceed 10,000 records and dominate model learning. In contrast, Anxiety, Bipolar, Stress, and Personality Disorder range from approximately 1,000 to 4,000 records. This imbalance was identified as the primary challenge for both models and addressed through class-weighted training.

**Missing Values**
A total of 362 rows had missing `statement` values. These were identified, verified (no empty strings or whitespace-only entries), and removed. The cleaned dataset contains 52,681 records.

**Text Characteristics**
Statements vary significantly in length. The average token count per statement after preprocessing was used to derive a padding length of approximately 208 tokens, which covers 96% of all statements without truncation.

**Label Distribution**
Seven distinct mental health sentiment categories are present. The encoding maps: Anxiety → 0, Bipolar → 1, Depression → 2, Normal → 3, Personality Disorder → 4, Stress → 5, Suicidal → 6.

---

## 💡 Key Findings

**Text Preprocessing**
Both models apply a consistent preprocessing pipeline: lowercasing, stopword removal (NLTK English stopwords), URL removal, special character and digit stripping, and whitespace normalization. Clean text is stored as a new column for reproducibility.

**Logistic Regression (TF-IDF)**
- Feature space: TF-IDF with up to 30,000 features using unigrams, bigrams, and trigrams
- Hyperparameter search over regularization strengths (`C`: 0.001, 0.01, 1) and solvers (`lbfgs`, `saga`) with `class_weight='balanced'` to handle imbalance
- Best model selected by highest validation accuracy
- Post-analysis separates correct and incorrect predictions by class for qualitative review

**LSTM (Word2Vec + RNN)**
- Custom Word2Vec embeddings (32-dim, skip-gram, window=5) trained on the training corpus
- Padding applied at max_tokens ≈ 208 (mean + 2 std), covering 96% of statements
- LSTM architecture: Embedding → SpatialDropout1D(0.3) → LSTM(64, dropout=0.4, recurrent_dropout=0.3) → Dropout(0.3) → Dense(7, softmax)
- Trained with `class_weight='balanced'`, EarlyStopping (patience=5), and ReduceLROnPlateau (patience=2, factor=0.5)
- Captures sequential context and word order, which TF-IDF cannot

**Class Imbalance Handling**
Both models use `class_weight='balanced'` to upweight minority classes (Stress, Personality Disorder, Anxiety) during training, preventing the model from defaulting to majority class predictions.

**Dominant vs Minority Class Performance**
Normal, Depression, and Suicidal classes consistently achieve higher per-class precision and recall due to their larger representation. Minority classes such as Stress and Personality Disorder remain harder to classify due to linguistic overlap with neighboring conditions.

---

## 💼 Business Impact

| Stakeholder | Impact |
|---|---|
| **Mental Health Platforms** | Automatically flag high-risk statements (Suicidal, Depression) for priority human review, reducing triage time |
| **Crisis Intervention Services** | Enable real-time screening of incoming messages or posts for suicidal ideation, supporting faster outreach |
| **Clinical Decision Support** | Augment clinician intake workflows by pre-labeling patient-reported text with likely sentiment categories |
| **Research & Policy** | Provide scalable tools for large-scale mental health trend analysis across populations and time periods |
| **Digital Health Apps** | Power in-app sentiment monitoring to surface personalized resources or escalate to human support when risk is detected |

A model that reliably distinguishes between Suicidal and Depression — or between Normal and Anxiety — can meaningfully change the speed and accuracy of mental health support delivery at scale.

---

## 📁 Repository Structure

```
Sentiment-Analysis-Mental-Health/
│
├── Data/
│   ├── sentiment_health.csv                                           # Raw dataset
│   └── sentiment_health_cleaned.csv                                   # Cleaned dataset (output of EDA notebook)
│
├── Notebooks/
│   ├── Sentiment_Analysis_for_Mental_Health___EDA.ipynb                        # Notebook 1: Data exploration & cleaning
│   ├── Sentiment_Analysis_for_Mental_Health___Logistic_Regression_Model.ipynb  # Notebook 2: TF-IDF + Logistic Regression
│   └── Sentiment_Analysis_for_Mental_Health___RNN_Model.ipynb                  # Notebook 3: Word2Vec + LSTM
│
└── README.md
```

---

## 👩‍💻 Author

**Reemika Subrata Das**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-reemikadas-blue?logo=linkedin&style=flat)](https://linkedin.com/in/reemikadas)
[![GitHub](https://img.shields.io/badge/GitHub-reemikadas-black?logo=github&style=flat)](https://github.com/reemikadas)
[![Email](https://img.shields.io/badge/Email-das.reemika%40gmail.com-red?logo=gmail&style=flat)](mailto:das.reemika@gmail.com)
