# BERT Sentiment Analysis

A Natural Language Processing (NLP) project that uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify customer reviews into three sentiment categories:

* Negative
* Neutral
* Positive

The project includes data preprocessing, exploratory data analysis, sentiment labeling, BERT tokenization, model training, evaluation, and a web-based interface for sentiment prediction.

## Project Overview

Customer reviews contain valuable information about user experiences and opinions. This project applies a transformer-based BERT model to automatically analyze review text and classify its sentiment.

The project was developed using customer reviews collected from multiple platforms and services, including Trustpilot reviews.

## Technologies Used

* Python
* BERT
* Hugging Face Transformers
* PyTorch
* Natural Language Processing (NLP)
* Pandas
* Scikit-learn
* HTML
* CSS
* JavaScript
* Flask

## Dataset

The final sentiment dataset contains:

* **Total Reviews:** 1,572
* **Training Samples:** 1,257
* **Testing Samples:** 315

### Sentiment Distribution

| Sentiment | Label |
| --------- | ----: |
| Negative  |     0 |
| Neutral   |     1 |
| Positive  |     2 |

## BERT Model

The project uses:

```text
bert-base-uncased
```

The model was implemented using Hugging Face's `BertForSequenceClassification` for three-class sentiment classification.

### Training Configuration

* Model: BERT Base Uncased
* Batch Size: 16
* Epochs: 3
* Training Environment: Google Colab
* GPU: Tesla T4
* Framework: PyTorch

## Model Performance

The trained BERT model achieved the following results on the test dataset:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 91.75% |
| Precision | 90.82% |
| Recall    | 91.75% |
| F1-Score  | 91.19% |

The model performed strongly overall, although the Neutral class had lower performance due to its smaller representation in the test data.

## Project Workflow

```text
Customer Reviews
       ↓
Data Collection
       ↓
Data Cleaning & Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Sentiment Label Creation
       ↓
BERT Tokenization
       ↓
BERT Model Training
       ↓
Model Evaluation
       ↓
Sentiment Prediction
       ↓
Web Application
```

## Project Structure

```text
Trustpilot-Sentiment-Analysis/
│
├── analysis/
│   ├── feature_engineering.py
│   ├── statistical_analysis.py
│   └── statistical_visualizations.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── sentiment_dataset.csv
│   ├── all_reviews_clean.csv
│   ├── transformer/
│   └── ...
│
├── eda/
│   ├── eda_analysis.py
│   └── text_analysis.py
│
├── models/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── preprocessing/
│   ├── clean_dataset.py
│   ├── create_sentiment_labels.py
│   ├── prepare_transformer_data.py
│   ├── tokenize_transformer.py
│   └── transformer_preprocessing.py
│
├── scraping/
│   ├── trustpilot_scraper.py
│   ├── temu_pagination_scraper.py
│   └── temu_scraper_backup.py
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

## Features

* Customer review sentiment classification
* Three-class sentiment prediction
* BERT-based text classification
* Text preprocessing and cleaning
* Exploratory data analysis
* Feature engineering
* Transformer tokenization
* Model evaluation
* Flask-based web interface
* Multiple review datasets

## Example

Input:

```text
The product quality is excellent and I am very satisfied.
```

Prediction:

```text
Positive
```

Input:

```text
The service was terrible and I am very disappointed.
```

Prediction:

```text
Negative
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ayesha-waris639/Trustpilot-Sentiment-Analysis.git
```

Navigate to the project:

```bash
cd Trustpilot-Sentiment-Analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
source venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Then open the local URL displayed by Flask in your browser.

## Skills Demonstrated

This project demonstrates practical experience in:

* Natural Language Processing
* Transformer Models
* BERT
* Sentiment Analysis
* Deep Learning
* Text Classification
* Data Preprocessing
* Exploratory Data Analysis
* Model Evaluation
* Python Development
* Flask Web Development

## Future Improvements

* Increase the size of the Neutral sentiment class
* Fine-tune the model with a larger dataset
* Improve class balance
* Deploy the application online
* Add real-time review analysis
* Add interactive analytics and visualizations

## Author

**Ayesha Waris**

AI/ML Enthusiast | NLP | Deep Learning | Generative AI

## GitHub

[Trustpilot Sentiment Analysis Repository] https://github.com/ayesha-waris639/Trustpilot-Sentiment-Analysis.git
