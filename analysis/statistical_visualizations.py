import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/sentiment_dataset.csv")

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("STATISTICAL VISUALIZATIONS")
print("=" * 60)

# --------------------------------------------------
# 1. Rating Distribution
# --------------------------------------------------

rating_counts = df["Rating"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
rating_counts.plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/12_rating_distribution.png")
plt.close()

print("Created: 12_rating_distribution.png")


# --------------------------------------------------
# 2. Average Rating by Company
# --------------------------------------------------

company_rating = df.groupby("Company")["Rating"].mean().sort_values()

plt.figure(figsize=(10, 6))
company_rating.plot(kind="bar")
plt.title("Average Rating by Company")
plt.xlabel("Company")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/13_average_rating_by_company.png")
plt.close()

print("Created: 13_average_rating_by_company.png")


# --------------------------------------------------
# 3. Sentiment Distribution
# --------------------------------------------------

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/14_sentiment_distribution.png")
plt.close()

print("Created: 14_sentiment_distribution.png")


# --------------------------------------------------
# 4. Sentiment by Company
# --------------------------------------------------

sentiment_company = pd.crosstab(
    df["Company"],
    df["Sentiment"]
)

sentiment_company.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Sentiment Distribution by Company")
plt.xlabel("Company")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("outputs/15_sentiment_by_company.png")
plt.close()

print("Created: 15_sentiment_by_company.png")


# --------------------------------------------------
# 5. Average Review Length by Company
# --------------------------------------------------

df["Review_Length"] = df["Review Text"].astype(str).str.len()

review_length = (
    df.groupby("Company")["Review_Length"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 6))
review_length.plot(kind="bar")
plt.title("Average Review Length by Company")
plt.xlabel("Company")
plt.ylabel("Average Characters")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/16_average_review_length.png")
plt.close()

print("Created: 16_average_review_length.png")


# --------------------------------------------------
# 6. Rating vs Sentiment
# --------------------------------------------------

rating_sentiment = pd.crosstab(
    df["Rating"],
    df["Sentiment"]
)

rating_sentiment.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Rating vs Sentiment")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("outputs/17_rating_vs_sentiment.png")
plt.close()

print("Created: 17_rating_vs_sentiment.png")


# --------------------------------------------------
# 7. Industry Average Rating
# --------------------------------------------------

industry_rating = (
    df.groupby("Industry")["Rating"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 6))
industry_rating.plot(kind="bar")
plt.title("Average Rating by Industry")
plt.xlabel("Industry")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/18_average_rating_by_industry.png")
plt.close()

print("Created: 18_average_rating_by_industry.png")


# --------------------------------------------------
# 8. Reviews by Company
# --------------------------------------------------

company_counts = df["Company"].value_counts().sort_values()

plt.figure(figsize=(10, 6))
company_counts.plot(kind="bar")
plt.title("Number of Reviews by Company")
plt.xlabel("Company")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/19_reviews_by_company.png")
plt.close()

print("Created: 19_reviews_by_company.png")


print("\n" + "=" * 60)
print("STATISTICAL VISUALIZATIONS COMPLETED")
print("=" * 60)

print("\nGraphs saved in: outputs/")