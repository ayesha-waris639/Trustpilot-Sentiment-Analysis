import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==========================================
# LOAD DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sentiment_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# READ DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df["Review Text"] = df["Review Text"].fillna("")

X = df["Review Text"]
y = df["Sentiment"]


# ==========================================
# SAME TRAIN/TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD MODEL AND TF-IDF
# ==========================================

model = joblib.load(MODEL_PATH)

tfidf = joblib.load(VECTORIZER_PATH)


# ==========================================
# TRANSFORM TEST DATA
# ==========================================

X_test_tfidf = tfidf.transform(X_test)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print(f"\nAccuracy Percentage: {accuracy * 100:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"]
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# CONFUSION MATRIX VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix - Sentiment Classification")

plt.xlabel("Predicted Sentiment")

plt.ylabel("Actual Sentiment")

plt.xticks(
    range(3),
    ["Negative", "Neutral", "Positive"]
)

plt.yticks(
    range(3),
    ["Negative", "Neutral", "Positive"]
)


# Add values inside matrix

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "20_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = os.path.join(
    OUTPUT_DIR,
    "model_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ==========================================
# COMPLETED
# ==========================================

print("\nEvaluation completed successfully.")

print("\nConfusion Matrix saved to:")
print(confusion_path)

print("\nClassification report saved to:")
print(report_path)