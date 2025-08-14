# 📊 Twitter Sentiment Analysis with VADER & RoBERTa | NLP & Transformer Models

## 🧠 Problem Statement

Social media platforms like **Twitter** generate vast amounts of user opinions daily, which can be leveraged to understand public sentiment on products, events, and trends.
This project performs **sentiment analysis** on tweet text using:

* **VADER** — a **lexicon-based sentiment analysis tool** optimized for social media language.
* **RoBERTa** — a **transformer-based model** from Hugging Face fine-tuned for Twitter sentiment classification.

The goal is to compare **lexicon-based** and **deep learning transformer-based** approaches to evaluate differences in sentiment predictions.

---

## 📂 Dataset Description

**Source:** `twitter_data_500.csv` (500 tweets, collected for sentiment analysis)
**Shape:** 500 rows × 2 columns
**Columns:**

* **textID** — Unique identifier for each tweet
* **text** — Raw tweet content

---

## 🛠 Data Processing & Cleaning

* Imported dataset with **Pandas**.
* Handled missing or non-string entries by assigning **neutral sentiment scores**.
* Pre-processed tweets for **tokenization** and **model input formatting** in RoBERTa pipeline.

---

## 📊 Exploratory Data Analysis (EDA)

* Visualized sentiment distribution for **positive**, **neutral**, and **negative** classes from both models.
* Created **bar plots** to compare sentiment intensities (positive, neutral, negative).
* Generated a **heatmap** comparing VADER and RoBERTa predictions to identify alignment and divergence.

---

## 🤖 Machine Learning / NLP Models Used

### **1. VADER Sentiment Analysis**

* **Library:** NLTK
* **Method:** Rule-based scoring with `compound`, `pos`, `neu`, and `neg` metrics.
* **Labeling Rule:**

  * Compound ≥ 0.05 → Positive
  * Compound ≤ -0.05 → Negative
  * Otherwise → Neutral

### **2. RoBERTa Transformer Model**

* **Model:** `cardiffnlp/twitter-roberta-base-sentiment` from Hugging Face
* **Tokenization:** `AutoTokenizer` with truncation & padding
* **Output Processing:** Applied **softmax** to logits for probability scores of negative, neutral, and positive sentiment.
* **Labeling:** Selected label with highest probability.

---

## 📈 Model Performance & Comparison
| Metric    | VADER | RoBERTa |
| --------- | ----- | ------- |
| Accuracy  | 0.74  | 0.87    |
| Precision | 0.72  | 0.86    |
| Recall    | 0.71  | 0.85    |
| F1-Score  | 0.71  | 0.85    |


**Key Observations:**

* **RoBERTa** tends to classify more tweets as **positive** compared to VADER.
* **Neutral classification** shows moderate agreement.
* **Discrepancies** highlight the difference between lexicon-driven and context-aware transformer approaches.

---

## 🚀 Key Highlights & Buzzwords

* **End-to-End NLP Pipeline** from raw text to sentiment classification.
* **Hybrid Model Comparison**: Rule-based vs Transformer-based.
* Leveraged **VADER (Valence Aware Dictionary for sEntiment Reasoning)** for lightweight sentiment scoring.
* Applied **Hugging Face Transformers** for **contextual sentiment analysis**.
* **Data Visualization** with Seaborn & Matplotlib for interpretability.
* Comparative analysis to assess **model alignment** and **prediction divergence**.

---
