import pandas as pd
import re
import string

# ==========================================
# TRUSTPILOT - FEATURE ENGINEERING
# ==========================================

INPUT_FILE = "data/transformer_preprocessed.csv"
OUTPUT_FILE = "data/feature_engineered.csv"

print("=" * 60)
print("FEATURE ENGINEERING - TRUSTPILOT SENTIMENT DATASET")
print("=" * 60)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nInitial Dataset Shape:")
print(df.shape)


# ------------------------------------------
# Helper functions
# ------------------------------------------

def word_count(text):
    return len(re.findall(r"\b\w+\b", str(text)))


def character_count(text):
    return len(str(text))


def sentence_count(text):
    text = str(text).strip()

    if not text:
        return 0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return len(sentences)


def average_word_length(text):
    words = re.findall(r"\b\w+\b", str(text))

    if not words:
        return 0

    return sum(len(word) for word in words) / len(words)


def uppercase_word_count(text):
    words = re.findall(r"\b[A-Za-z]+\b", str(text))

    return sum(1 for word in words if word.isupper() and len(word) > 1)


def punctuation_count(text):
    return sum(1 for char in str(text) if char in string.punctuation)


def exclamation_count(text):
    return str(text).count("!")


def question_count(text):
    return str(text).count("?")


# ------------------------------------------
# Create required features
# ------------------------------------------

print("\nCreating text features...")

df["Word Count"] = df["Combined Text"].apply(word_count)

df["Character Count"] = df["Combined Text"].apply(character_count)

df["Sentence Count"] = df["Combined Text"].apply(sentence_count)

df["Average Word Length"] = df["Combined Text"].apply(
    average_word_length
)

df["Uppercase Word Count"] = df["Combined Text"].apply(
    uppercase_word_count
)

df["Punctuation Count"] = df["Combined Text"].apply(
    punctuation_count
)

df["Exclamation Count"] = df["Combined Text"].apply(
    exclamation_count
)

df["Question Count"] = df["Combined Text"].apply(
    question_count
)


# ------------------------------------------
# Display results
# ------------------------------------------

print("\nFeature Engineering Completed")

print("\nNew Features:")
print("""
1. Word Count
2. Character Count
3. Sentence Count
4. Average Word Length
5. Uppercase Word Count
6. Punctuation Count
7. Exclamation Count
8. Question Count
""")

print("\nAverage Features by Sentiment:")

features = [
    "Word Count",
    "Character Count",
    "Sentence Count",
    "Average Word Length",
    "Uppercase Word Count",
    "Punctuation Count",
    "Exclamation Count",
    "Question Count"
]

print(df.groupby("Sentiment")[features].mean().round(2))


# ------------------------------------------
# Save dataset
# ------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY")