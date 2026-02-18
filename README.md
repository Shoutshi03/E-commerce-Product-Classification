# E-Commerce Product Classification using ML & NLP

## Project Overview

This project aims to automatically classify e-commerce products into categories
based on their **title** and **description** using classical NLP techniques.

Three machine learning models were compared:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC

The best model achieved over **97.22% accuracy**.

## Business Problem

E-commerce platforms manage thousands of products daily.  
Manual categorization is:

- Time-consuming
- Error-prone
- Expensive

This system automates product classification to improve scalability and efficiency.

## Dataset

[lien du dataset](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

- Text-based dataset (product title + description)
- Multi-class classification
- Preprocessing includes:
  - Text cleaning
  - TF-IDF vectorization

## Methodology

Pipeline:

Raw Text  
↓  
TF-IDF Vectorization  
↓  
ML Model (LR / NB / SVC)  
↓  
Evaluation  
↓  
Best Model Selection  

---

## Model Comparison

![Model Comparison](imgs/models_comparaison.png)

| Model | Accuracy | F1-Score |

|--------|----------|----------|

| Logistic Regression | 88% | 87% |

| Multinomial NB | 85% | 84% |

| Linear SVC | 97.22% | 97.22% |

---

## Confusion Matrix (Best Model)

![Confusion Matrix](imgs/confusion_matrix.png)

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Joblib

---

## How to Run

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
python src/train.py
python src/evaluate.py
