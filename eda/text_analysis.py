import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import os

# Load dataset
df = pd.read_csv(
    "data/sentiment_dataset.csv",
    encoding="utf-8-sig"
)

os.makedirs("outputs", exist_ok=True)

# Basic English stopwords
stopwords = {
    "the", "and", "to", "of", "a", "in", "is", "it", "for",
    "on", "that", "this", "was", "with", "my", "but", "they",
    "i", "you", "we", "are", "be", "have", "had", "not",
    "from", "as", "at", "or", "an", "so", "very", "their",
    "me", "your", "can", "will", "would", "just", "been",
    "all", "our", "there", "were", "what", "when", "if",
    "about", "get", "got", "has", "do", "did", "up", "out",
    "no", "more", "one", "they", "them", "he", "she"
}


def clean_words(text):
    text = str(text).lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)

    words = [
        word for word in words
        if word not in stopwords
    ]

    return words


# ==============================
# COMMON WORDS - ALL REVIEWS
# ==============================

all_words = []

for text in df["Review Text"]:
    all_words.extend(clean_words(text))

word_counts = Counter(all_words)

print("=" * 60)
print("TOP 20 MOST COMMON WORDS")
print("=" * 60)

for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")


# ==============================
# POSITIVE WORDS
# ==============================

positive_words = []

for text in df.loc[
    df["Sentiment"] == "Positive",
    "Review Text"
]:
    positive_words.extend(clean_words(text))

positive_counts = Counter(positive_words)

print("\n" + "=" * 60)
print("TOP 20 POSITIVE REVIEW WORDS")
print("=" * 60)

for word, count in positive_counts.most_common(20):
    print(f"{word}: {count}")


# ==============================
# NEGATIVE WORDS
# ==============================

negative_words = []

for text in df.loc[
    df["Sentiment"] == "Negative",
    "Review Text"
]:
    negative_words.extend(clean_words(text))

negative_counts = Counter(negative_words)

print("\n" + "=" * 60)
print("TOP 20 NEGATIVE REVIEW WORDS")
print("=" * 60)

for word, count in negative_counts.most_common(20):
    print(f"{word}: {count}")


# ==============================
# WORD CLOUD - ALL REVIEWS
# ==============================

all_text = " ".join(
    df["Review Text"].astype(str)
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords,
    min_font_size=10
).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - All Reviews")
plt.tight_layout()

plt.savefig(
    "outputs/09_wordcloud_all.png",
    dpi=150
)

plt.close()


# ==============================
# WORD CLOUD - POSITIVE
# ==============================

positive_text = " ".join(
    df.loc[
        df["Sentiment"] == "Positive",
        "Review Text"
    ].astype(str)
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords,
    min_font_size=10
).generate(positive_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Positive Reviews")
plt.tight_layout()

plt.savefig(
    "outputs/10_wordcloud_positive.png",
    dpi=150
)

plt.close()


# ==============================
# WORD CLOUD - NEGATIVE
# ==============================

negative_text = " ".join(
    df.loc[
        df["Sentiment"] == "Negative",
        "Review Text"
    ].astype(str)
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords,
    min_font_size=10
).generate(negative_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Negative Reviews")
plt.tight_layout()

plt.savefig(
    "outputs/11_wordcloud_negative.png",
    dpi=150
)

plt.close()


print("\n" + "=" * 60)
print("TEXT ANALYSIS COMPLETED")
print("=" * 60)

print("Generated files:")
print("09_wordcloud_all.png")
print("10_wordcloud_positive.png")
print("11_wordcloud_negative.png")