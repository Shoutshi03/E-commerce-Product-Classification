# ==========================================================
# train.py - FINAL VERSION (Compatible with your dataset)
# Dataset format:
# Column 0 -> Category
# Column 1 -> Description
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = "../data/ecommerceDataset.csv"
MODEL_DIR = "../models"
IMG_DIR = "../imgs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ==========================================================
# LOAD DATASET (NO HEADER)
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, header=None)

# Rename columns manually
df.columns = ["Category", "Description"]

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
print(f"Number of classes: {df['Category'].nunique()}")

# ==========================================================
# PREPARE DATA
# ==========================================================

df = df.dropna()

X = df["Description"].astype(str)
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# DEFINE MODELS
# ==========================================================

MODELS = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Multinomial NB": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", MultinomialNB())
    ]),
    "Linear SVC": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", LinearSVC(random_state=42))
    ])
}

# ==========================================================
# TRAIN & COMPARE MODELS
# ==========================================================

model_names = []
accuracy_scores = []
f1_scores = []

best_model = None
best_f1 = 0
best_model_name = ""

print("\nTraining models...\n")

for name, pipeline in MODELS.items():

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average="weighted") * 100

    print(f"{name} → Accuracy: {acc:.2f}% | F1-score: {f1:.2f}%")

    model_names.append(name)
    accuracy_scores.append(acc)
    f1_scores.append(f1)

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# ==========================================================
# SAVE BEST MODEL
# ==========================================================

model_path = os.path.join(MODEL_DIR, "best_model.pkl")
joblib.dump(best_model, model_path)

print(f"\nBest model: {best_model_name}")
print(f"Model saved at: {model_path}")

# ==========================================================
# SAVE COMPARISON CHART
# ==========================================================

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, accuracy_scores, width, label="Accuracy (%)")
plt.bar(x + width/2, f1_scores, width, label="F1-score (%)")

plt.xticks(x, model_names)
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.title("Model Performance Comparison")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "model_comparison.png"))
plt.close()

print("Model comparison chart saved.")

print("\nTraining completed successfully.")
