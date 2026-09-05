import pandas as pd

input_file = "data/all_reviews_clean.csv"
output_file = "data/sentiment_dataset.csv"

# Load cleaned dataset
df = pd.read_csv(input_file, encoding="utf-8-sig")

# Convert rating into sentiment
def get_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["Sentiment"] = df["Rating"].apply(get_sentiment)

# Save sentiment dataset
df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print("SENTIMENT LABELING COMPLETED")
print("=" * 50)

print("Total reviews:", len(df))

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nSentiment percentages:")
print(
    (df["Sentiment"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nSaved to:", output_file)