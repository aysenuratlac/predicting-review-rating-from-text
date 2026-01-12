# Predicting Review Ratings from Turkish Text: A Comparative Study

This repository contains the source code and experimental framework for a systematic study on predicting user ratings from **Turkish Google Maps reviews**. 
The project was developed as a **graduation thesis** at **Istanbul Technical University (ITU)**, Department of Mathematical Engineering.

---

## 📌 Project Overview

The primary objective of this research is to **predict user ratings (1–5 stars)** directly from textual reviews. Since Turkish is a **morphologically rich and agglutinative language**, the study places special emphasis on how different **text representation and tokenization strategies** influence classification performance.

---

## 🔍 Key Research Questions

* How do different **tokenization levels** (word, syllable, character, lemma) impact sentiment-related tasks in Turkish?
* How does **class imbalance** affect model reliability (the *Accuracy Paradox*)?
* Does **morphological normalization** (lemmatization) improve or hinder rating prediction performance?

---

## 🚀 Features

* **Multi-Level Tokenization**

  * Word-level (NLTK)
  * Syllable-level (TurkishNLP)
  * Character-level (n-grams)
  * Lemmatization (Zeyrek Morphological Analyzer)

* **Pipeline Architecture**
  Integrated **Scikit-learn pipelines** for seamless vectorization and classification.

* **Imbalance Handling**
  Cost-sensitive learning via **class weighting** and **downsampling** strategies.

* **Statistical Validation**
  Bootstrap-based **95% Confidence Interval (CI)** estimation for all evaluation metrics.

* **Outlier Analysis**
  Percentile-based removal of text-length outliers to improve model robustness.

---

## 📂 Project Structure

```text
├── src/
│   └── tokenizers/
│       ├── word_tokenizer.py        # NLTK-based Turkish word tokenizer
│       ├── syllable_tokenizer.py    # Syllabification using TurkishNLP
│       └── zeyrek_word_tokenizer.py # Lemmatization using Zeyrek
├── data/
│   └── raw/                         # Public dataset (838k reviews)
├── experiment.py                    # Main experimental pipeline script
└── results/                         # JSON results and performance logs
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/turkish-review-prediction.git
cd turkish-review-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn nltk zeyrek turkishnlp matplotlib seaborn scipy
```

### 3. Download NLTK resources

```python
import nltk
nltk.download('punkt')
```

---

## 💻 Usage

The `experiment.py` script supports multiple experimental configurations through command-line arguments.

## Dataset

This repository does not include the raw dataset file due to GitHub file size limits.

Download the dataset from:
- <https://www.kaggle.com/datasets/abdullahkocak/gmaps800kreviews>

After downloading, place it here:
- `data/raw/public_dataset.csv`

### Example Commands

**Logistic Regression with Word-level Tokenization and TF-IDF**

```bash
python experiment.py \
  --model_type logistic \
  --vec_type tfidf \
  --token_type word \
  --exp exp_01
```

**SVM with Syllable-level Tokenization and Class Weighting**

```bash
python experiment.py \
  --model_type svm \
  --vec_type count \
  --token_type syllable \
  --class_weight balanced \
  --exp exp_02
```

**Logistic Regression with Character-level Tokenization and Downsampling**

```bash
python experiment.py \
  --model_type logistic \
  --vec_type tfidf \
  --token_type char \
  --downsample \
  --exp exp_03
```

---

## 📊 Summary of Findings

* **The Accuracy Paradox**
  In highly imbalanced datasets, accuracy reached up to **0.795**, yet this metric was misleading due to dominance of majority classes. **Macro-F1 (~0.50)** provided a more reliable evaluation.

* **Information Loss via Lemmatization**
  Lemmatization using **Zeyrek** consistently underperformed compared to raw word-level representations. In Turkish, suffixes encode crucial semantic and sentiment-related information; removing them leads to significant information loss.

* **Model Performance**
  **Logistic Regression** outperformed KNN and SVM in terms of stability, predictive accuracy, and computational efficiency.

* **Tokenization Effects**
  **Character-level** and **raw word-level** tokenization produced the most discriminative feature sets for rating prediction.

---

## 🎓 Citation

If you find this work useful, please cite:

> Atlaç, A. (2026). *Predicting Review Ratings from Text: A Comparative Study of Tokenization, Vectorization, and Machine Learning Methods*. Final Thesis, Istanbul Technical University.


---

⭐ Contributions, issues, and feedback are welcome.
