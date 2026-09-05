from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# ==============================
# MODEL CONFIGURATION
# ==============================

MODEL_PATH = "bert_sentiment_final"

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# ==============================
# LOAD TOKENIZER
# ==============================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

# ==============================
# LOAD BERT MODEL
# ==============================

print("Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==============================
# PREDICTION API
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        text = data.get("text", "").strip()

        if not text:

            return jsonify({
                "success": False,
                "error": "Please enter some text."
            }), 400

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        # Move to device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Prediction
        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

        sentiment = LABELS[prediction]

        confidence = (
            probabilities[0, prediction].item()
            * 100
        )

        return jsonify({

            "success": True,

            "text": text,

            "sentiment": sentiment,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities": {

                "Negative": round(
                    probabilities[0, 0].item() * 100,
                    2
                ),

                "Neutral": round(
                    probabilities[0, 1].item() * 100,
                    2
                ),

                "Positive": round(
                    probabilities[0, 2].item() * 100,
                    2
                )
            }
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
    