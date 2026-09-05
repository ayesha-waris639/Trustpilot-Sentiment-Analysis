import pandas as pd
import os

# Load dataset
file_path = "data/sentiment_dataset.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("STATISTICAL ANALYSIS - TRUSTPILOT SENTIMENT DATASET")
print("=" * 60)

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------------
# 1. Rating Statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("1. RATING STATISTICS")
print("=" * 60)

print("\nMean Rating:")
print(round(df["Rating"].mean(), 2))

print("\nMedian Rating:")
print(df["Rating"].median())

print("\nMode Rating:")
print(df["Rating"].mode()[0])

print("\nStandard Deviation:")
print(round(df["Rating"].std(), 2))

# --------------------------------------------------
# 2. Rating Distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("2. RATING DISTRIBUTION")
print("=" * 60)

rating_distribution = df["Rating"].value_counts().sort_index()

print(rating_distribution)

# --------------------------------------------------
# 3. Sentiment Distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("3. SENTIMENT DISTRIBUTION")
print("=" * 60)

sentiment_distribution = df["Sentiment"].value_counts()

print(sentiment_distribution)

# --------------------------------------------------
# 4. Company Statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("4. COMPANY STATISTICS")
print("=" * 60)

company_stats = df.groupby("Company").agg(
    Total_Reviews=("Company", "count"),
    Average_Rating=("Rating", "mean")
)

company_stats["Average_Rating"] = company_stats["Average_Rating"].round(2)

print(company_stats)

# --------------------------------------------------
# 5. Sentiment by Company
# --------------------------------------------------

print("\n" + "=" * 60)
print("5. SENTIMENT BY COMPANY")
print("=" * 60)

sentiment_company = pd.crosstab(
    df["Company"],
    df["Sentiment"]
)

print(sentiment_company)

# --------------------------------------------------
# 6. Sentiment Percentage by Company
# --------------------------------------------------

print("\n" + "=" * 60)
print("6. SENTIMENT PERCENTAGE BY COMPANY")
print("=" * 60)

sentiment_percentage = pd.crosstab(
    df["Company"],
    df["Sentiment"],
    normalize="index"
) * 100

print(sentiment_percentage.round(2))

# --------------------------------------------------
# 7. Review Length Statistics
# --------------------------------------------------

df["Review_Length"] = df["Review Text"].astype(str).str.len()

print("\n" + "=" * 60)
print("7. REVIEW LENGTH STATISTICS")
print("=" * 60)

print("\nAverage Review Length:")
print(round(df["Review_Length"].mean(), 2))

print("\nMedian Review Length:")
print(round(df["Review_Length"].median(), 2))

print("\nLongest Review:")
print(df["Review_Length"].max())

print("\nShortest Review:")
print(df["Review_Length"].min())

# --------------------------------------------------
# 8. Industry Statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("8. INDUSTRY STATISTICS")
print("=" * 60)

industry_stats = df.groupby("Industry").agg(
    Total_Reviews=("Industry", "count"),
    Average_Rating=("Rating", "mean")
)

industry_stats["Average_Rating"] = industry_stats["Average_Rating"].round(2)

print(industry_stats)

# --------------------------------------------------
# Save results
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

company_stats.to_csv(
    "outputs/company_statistics.csv"
)

sentiment_company.to_csv(
    "outputs/sentiment_by_company.csv"
)

sentiment_percentage.to_csv(
    "outputs/sentiment_percentage_by_company.csv"
)

industry_stats.to_csv(
    "outputs/industry_statistics.csv"
)

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS COMPLETED")
print("=" * 60)

print("\nResults saved in outputs/")