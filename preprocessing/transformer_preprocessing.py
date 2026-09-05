import pandas as pd
import re
import os
import html


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sentiment_dataset.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "transformer_preprocessed.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("TRANSFORMER TEXT PREPROCESSING")
print("=" * 60)

print("\nInitial dataset shape:")
print(df.shape)


# ==========================================
# STEP 1
# MERGE REVIEW TITLE + REVIEW TEXT
# ==========================================

df["Review Title"] = df["Review Title"].fillna("")
df["Review Text"] = df["Review Text"].fillna("")

df["Combined Text"] = (
    df["Review Title"].astype(str).str.strip()
    + " "
    + df["Review Text"].astype(str).str.strip()
)

print("\nStep 1: Review Title + Review Text merged")


# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    text = str(text)

    # STEP 2: Lowercase
    text = text.lower()

    # STEP 3: Remove HTML tags
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # STEP 4: Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # STEP 5: Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text)

    # STEP 6: Remove irrelevant special characters
    # Keep letters, numbers and meaningful punctuation
    text = re.sub(
        r"[^a-z0-9\s.,!?'\-]",
        " ",
        text
    )

    # STEP 7: Expand common contractions

    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "mustn't": "must not",
        "i'm": "i am",
        "i've": "i have",
        "i'll": "i will",
        "i'd": "i would",
        "you're": "you are",
        "you've": "you have",
        "you'll": "you will",
        "you'd": "you would",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "we've": "we have",
        "we'll": "we will",
        "they're": "they are",
        "they've": "they have",
        "they'll": "they will"
    }

    for contraction, expansion in contractions.items():
        text = text.replace(
            contraction,
            expansion
        )

    # STEP 8: Remove emojis / unsupported symbols
    # Already handled by special-character filtering above

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# APPLY TEXT CLEANING
# ==========================================

df["Combined Text"] = df["Combined Text"].apply(
    clean_text
)

print("Steps 2-8: Text cleaning completed")


# ==========================================
# STEP 9: REMOVE DUPLICATES
# ==========================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["Combined Text"]
)

duplicates_removed = (
    before_duplicates - len(df)
)

print(
    f"\nStep 9: Duplicate reviews removed: "
    f"{duplicates_removed}"
)


# ==========================================
# STEP 10: REMOVE EMPTY REVIEWS
# ==========================================

before_empty = len(df)

df = df[
    df["Combined Text"].str.strip() != ""
]

empty_removed = (
    before_empty - len(df)
)

print(
    f"Step 10: Empty reviews removed: "
    f"{empty_removed}"
)


# ==========================================
# IMPORTANT TRANSFORMER RULES
# ==========================================

print("\nTransformer preprocessing rules:")
print("Stopwords removed: NO")
print("Stemming applied: NO")
print("Lemmatization applied: NO")


# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# FINAL INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(list(df.columns))

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())