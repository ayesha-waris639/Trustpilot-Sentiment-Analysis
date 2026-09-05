import os
import pandas as pd
from transformers import AutoTokenizer

# ==============================
# PATHS
# ==============================

TRAIN_PATH = "data/transformer/train.csv"
TEST_PATH = "data/transformer/test.csv"

OUTPUT_DIR = "data/transformer/tokenized"

# ==============================
# BERT TOKENIZER
# ==============================

MODEL_NAME = "bert-base-uncased"

print("Loading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")

# ==============================
# LOAD DATA
# ==============================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTraining data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# ==============================
# TEXT COLUMN
# ==============================

TEXT_COLUMN = "Combined Text"

# Check required column
if TEXT_COLUMN not in train_df.columns:
    raise ValueError(
        f"Column '{TEXT_COLUMN}' not found in training data. "
        f"Available columns: {list(train_df.columns)}"
    )

if TEXT_COLUMN not in test_df.columns:
    raise ValueError(
        f"Column '{TEXT_COLUMN}' not found in testing data. "
        f"Available columns: {list(test_df.columns)}"
    )

# ==============================
# TOKENIZATION
# ==============================

MAX_LENGTH = 128

print("\nTokenizing training data...")

train_encodings = tokenizer(
    train_df[TEXT_COLUMN].astype(str).tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH
)

print("Training tokenization complete!")

print("\nTokenizing testing data...")

test_encodings = tokenizer(
    test_df[TEXT_COLUMN].astype(str).tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH
)

print("Testing tokenization complete!")

# ==============================
# CREATE OUTPUT DIRECTORY
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# SAVE TOKENIZED DATA
# ==============================

train_output = {
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "label": train_df["label"].tolist()
}

test_output = {
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "label": test_df["label"].tolist()
}

# Save as PyTorch tensors
import torch

torch.save(
    train_output,
    os.path.join(OUTPUT_DIR, "train_tokenized.pt")
)

torch.save(
    test_output,
    os.path.join(OUTPUT_DIR, "test_tokenized.pt")
)

# ==============================
# SAVE TOKENIZER
# ==============================

tokenizer.save_pretrained(
    os.path.join(OUTPUT_DIR, "tokenizer")
)

# ==============================
# RESULTS
# ==============================

print("\n" + "=" * 50)
print("TOKENIZATION COMPLETE")
print("=" * 50)

print("Training samples:", len(train_output["input_ids"]))
print("Testing samples:", len(test_output["input_ids"]))

print("Maximum sequence length:", MAX_LENGTH)

print(
    "Training input_ids shape:",
    (len(train_output["input_ids"]), len(train_output["input_ids"][0]))
)

print(
    "Testing input_ids shape:",
    (len(test_output["input_ids"]), len(test_output["input_ids"][0]))
)

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. train_tokenized.pt")
print("2. test_tokenized.pt")
print("3. tokenizer/")