"""
preprocess.py

This module handles text preprocessing and feature extraction
for the E-commerce Product Classification project.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ----------------------------
# 1 -Text Cleaning Function
# ----------------------------

def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Removing extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove numbers
    text = re.sub(r"\s+", " ", text)     # Remove extra spaces

    return text.strip()


# ----------------------------
# 2 -Load and Prepare Dataset
# ----------------------------

def load_and_prepare_data(file_path: str):
    """
    Loads dataset and applies basic cleaning.
    Expected columns:
    - Category
    - Description
    """

    df = pd.read_csv(file_path)

    # Rename columns if needed
    df.columns = ["Category", "Description"]

    # Drop missing values
    df = df.dropna()

    # Clean text
    df["clean_text"] = df["Description"].apply(clean_text)

    return df


# ----------------------------
# 3 -Train-Test Split
# ----------------------------

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits dataset into training and testing sets.
    """

    X = df["clean_text"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ----------------------------
# 4 -TF-IDF Vectorization
# ----------------------------

def vectorize_text(X_train, X_test, max_features=5000):
    """
    Converts text into TF-IDF features.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer


# ----------------------------
# 5 -Full Preprocessing Pipeline
# ----------------------------

def preprocess_pipeline(file_path: str):
    """
    Complete preprocessing pipeline:
    - Load data
    - Clean text
    - Split dataset
    - Apply TF-IDF
    """

    df = load_and_prepare_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# ----------------------------
# 6 -Script Execution (Optional)
# ----------------------------

if __name__ == "__main__":
    file_path = "../data/ecommerceDataset.csv"

    X_train, X_test, y_train, y_test, vectorizer = preprocess_pipeline(file_path)

    print("Preprocessing completed successfully.")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
