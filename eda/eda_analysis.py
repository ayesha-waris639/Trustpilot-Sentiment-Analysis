import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# LOAD DATASET
# ==============================

file_path = "data/sentiment_dataset.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TRUSTPILOT SENTIMENT ANALYSIS - EDA")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==============================
# 1. SENTIMENT DISTRIBUTION
# ==============================

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar")
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/01_sentiment_distribution.png")
plt.close()


# ==============================
# 2. RATING DISTRIBUTION
# ==============================

rating_counts = df["Rating"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
rating_counts.plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/02_rating_distribution.png")
plt.close()


# ==============================
# 3. COMPANY REVIEW COUNT
# ==============================

company_counts = df["Company"].value_counts()

plt.figure(figsize=(10, 6))
company_counts.plot(kind="bar")
plt.title("Reviews Collected by Company")
plt.xlabel("Company")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/03_company_review_count.png")
plt.close()


# ==============================
# 4. COMPANY VS SENTIMENT
# ==============================

company_sentiment = pd.crosstab(
    df["Company"],
    df["Sentiment"]
)

company_sentiment.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Sentiment Distribution by Company")
plt.xlabel("Company")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/04_company_sentiment.png")
plt.close()


# ==============================
# 5. INDUSTRY VS SENTIMENT
# ==============================

industry_sentiment = pd.crosstab(
    df["Industry"],
    df["Sentiment"]
)

industry_sentiment.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Sentiment Distribution by Industry")
plt.xlabel("Industry")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/05_industry_sentiment.png")
plt.close()


# ==============================
# 6. AVERAGE RATING BY COMPANY
# ==============================

average_rating = (
    df.groupby("Company")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
average_rating.plot(kind="bar")

plt.title("Average Rating by Company")
plt.xlabel("Company")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 5)

plt.tight_layout()

plt.savefig("outputs/06_average_rating_company.png")
plt.close()


# ==============================
# 7. SENTIMENT PERCENTAGE BY COMPANY
# ==============================

sentiment_percentage = pd.crosstab(
    df["Company"],
    df["Sentiment"],
    normalize="index"
) * 100

sentiment_percentage.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Sentiment Percentage by Company")
plt.xlabel("Company")
plt.ylabel("Percentage")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Sentiment")

plt.tight_layout()

plt.savefig("outputs/07_sentiment_percentage_company.png")
plt.close()


# ==============================
# 8. REVIEW LENGTH
# ==============================

df["Review_Length"] = df["Review Text"].astype(str).str.len()

plt.figure(figsize=(8, 5))
plt.hist(df["Review_Length"], bins=30)

plt.title("Review Text Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("outputs/08_review_length_distribution.png")
plt.close()


# ==============================
# SUMMARY
# ==============================

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

print("\nTotal Reviews:", len(df))

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

print("\nAverage Rating:")
print(round(df["Rating"].mean(), 2))

print("\nAverage Rating by Company:")
print(average_rating.round(2))

print("\nAverage Review Length:")
print(round(df["Review_Length"].mean(), 2))

print("\nGraphs saved in:")
print("outputs/")

print("\nEDA COMPLETED SUCCESSFULLY")