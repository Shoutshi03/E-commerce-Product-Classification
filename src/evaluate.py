# ==========================================================
# evaluate.py - FINAL VERSION (Compatible with your dataset)
# ==========================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = "../data/ecommerceDataset.csv"
MODEL_PATH = "../models/best_model.pkl"
IMG_DIR = "../imgs"

os.makedirs(IMG_DIR, exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# ==========================================================
# LOAD DATASET (NO HEADER)
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, header=None)
df.columns = ["Category", "Description"]

df = df.dropna()

X = df["Description"].astype(str)
y = df["Category"]

# IMPORTANT: Same split as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating model...\n")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average="weighted") * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"F1-score: {f1:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Best Model")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_evaluation.png"))
plt.close()

print("\nConfusion matrix saved successfully.")
print("Evaluation completed.")
