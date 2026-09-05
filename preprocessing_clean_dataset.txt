import pandas as pd
import re
import os

# Load raw dataset
input_file = "data/all_reviews_raw.csv"
output_file = "data/all_reviews_clean.csv"

df = pd.read_csv(input_file, encoding="utf-8-sig")

print("=" * 50)
print("INITIAL DATASET")
print("=" * 50)
print("Rows:", len(df))
print("Columns:", list(df.columns))

# Remove completely duplicate reviews
before = len(df)

df = df.drop_duplicates(
    subset=["Company", "Review Text"],
    keep="first"
)

duplicates_removed = before - len(df)

# Clean Review Text
df["Review Text"] = (
    df["Review Text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Clean Review Title
df["Review Title"] = (
    df["Review Title"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Convert Rating to numeric
df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

# Keep only valid ratings
df = df[df["Rating"].between(1, 5)]

# Remove rows with empty review text
df = df[df["Review Text"].str.len() > 0]

# Reset index
df = df.reset_index(drop=True)

# Create output directory if needed
os.makedirs("data", exist_ok=True)

# Save cleaned dataset
df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 50)
print("CLEANING COMPLETED")
print("=" * 50)
print("Duplicates removed:", duplicates_removed)
print("Final reviews:", len(df))
print("Saved to:", output_file)

print("\nReviews by company:")
print(df.groupby("Company").size())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRating distribution:")
print(df["Rating"].value_counts().sort_index())