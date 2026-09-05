import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/transformer_preprocessed.csv"
OUTPUT_DIR = "data/transformer"

print("=" * 60)
print("TRANSFORMER DATA PREPARATION")
print("=" * 60)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nInitial Dataset Shape:")
print(df.shape)

# Keep required columns
df = df[["Combined Text", "Sentiment"]].copy()

# Remove missing values
df = df.dropna(subset=["Combined Text", "Sentiment"])

# Remove empty reviews
df = df[df["Combined Text"].str.strip() != ""]

print("\nDataset after cleaning:")
print(df.shape)

# Encode sentiment labels
label_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

df["label"] = df["Sentiment"].map(label_mapping)

print("\nLabel Mapping:")
print(label_mapping)

# Train/Test split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Save datasets
train_df.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining label distribution:")
print(train_df["Sentiment"].value_counts())

print("\nTesting label distribution:")
print(test_df["Sentiment"].value_counts())

print("\nSaved files:")
print(f"{OUTPUT_DIR}/train.csv")
print(f"{OUTPUT_DIR}/test.csv")

print("\nTRANSFORMER DATA PREPARATION COMPLETED SUCCESSFULLY")