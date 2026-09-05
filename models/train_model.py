import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "sentiment_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TRUSTPILOT SENTIMENT CLASSIFICATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())


# ==========================================
# 2. REMOVE EMPTY REVIEWS
# ==========================================

df["Review Text"] = df["Review Text"].fillna("")

df = df[df["Review Text"].str.strip() != ""]

print("\nDataset after removing empty reviews:")
print(df.shape)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df["Review Text"]
y = df["Sentiment"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. TF-IDF VECTORIZATION
# ==========================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ==========================================
# 6. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. SAVE MODEL
# ==========================================

MODEL_DIR = os.path.dirname(__file__)

model_path = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(model, model_path)
joblib.dump(tfidf, vectorizer_path)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("\nModel:")
print(model_path)

print("\nTF-IDF Vectorizer:")
print(vectorizer_path)

print("\nMachine Learning training completed successfully.")