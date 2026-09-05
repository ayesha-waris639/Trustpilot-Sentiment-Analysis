import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Trustpilot Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ==============================
# LOAD DATA
# ==============================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "sentiment_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

# ==============================
# TITLE
# ==============================

st.title("Trustpilot Sentiment Analysis Dashboard")
st.markdown(
    "Interactive analysis of customer reviews collected from Trustpilot."
)

# ==============================
# SIDEBAR
# ==============================

st.sidebar.header("Filters")

companies = ["All"] + sorted(df["Company"].unique().tolist())

selected_company = st.sidebar.selectbox(
    "Select Company",
    companies
)

sentiments = ["All"] + sorted(df["Sentiment"].unique().tolist())

selected_sentiment = st.sidebar.selectbox(
    "Select Sentiment",
    sentiments
)

# ==============================
# APPLY FILTERS
# ==============================

filtered_df = df.copy()

if selected_company != "All":
    filtered_df = filtered_df[
        filtered_df["Company"] == selected_company
    ]

if selected_sentiment != "All":
    filtered_df = filtered_df[
        filtered_df["Sentiment"] == selected_sentiment
    ]

# ==============================
# KPI SECTION
# ==============================

total_reviews = len(filtered_df)

average_rating = filtered_df["Rating"].mean()

positive_reviews = (
    filtered_df["Sentiment"] == "Positive"
).sum()

negative_reviews = (
    filtered_df["Sentiment"] == "Negative"
).sum()

neutral_reviews = (
    filtered_df["Sentiment"] == "Neutral"
).sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Reviews",
    total_reviews
)

col2.metric(
    "Average Rating",
    f"{average_rating:.2f}"
)

col3.metric(
    "Positive",
    positive_reviews
)

col4.metric(
    "Negative",
    negative_reviews
)

col5.metric(
    "Neutral",
    neutral_reviews
)

st.divider()

# ==============================
# CHART 1: SENTIMENT DISTRIBUTION
# ==============================

col1, col2 = st.columns(2)

with col1:

    sentiment_counts = (
        filtered_df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution",
        hole=0.4
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )

# ==============================
# CHART 2: RATING DISTRIBUTION
# ==============================

with col2:

    rating_counts = (
        filtered_df["Rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_counts.columns = [
        "Rating",
        "Count"
    ]

    fig_rating = px.bar(
        rating_counts,
        x="Rating",
        y="Count",
        title="Rating Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )

# ==============================
# COMPANY ANALYSIS
# ==============================

st.subheader("Company Analysis")

company_stats = (
    filtered_df
    .groupby("Company")
    .agg(
        Reviews=("Company", "size"),
        Average_Rating=("Rating", "mean")
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:

    fig_company_reviews = px.bar(
        company_stats,
        x="Company",
        y="Reviews",
        title="Reviews by Company",
        text="Reviews"
    )

    st.plotly_chart(
        fig_company_reviews,
        use_container_width=True
    )

with col2:

    fig_company_rating = px.bar(
        company_stats,
        x="Company",
        y="Average_Rating",
        title="Average Rating by Company",
        text="Average_Rating"
    )

    fig_company_rating.update_traces(
        texttemplate="%{text:.2f}"
    )

    st.plotly_chart(
        fig_company_rating,
        use_container_width=True
    )

# ==============================
# SENTIMENT BY COMPANY
# ==============================

st.subheader("Sentiment by Company")

sentiment_company = (
    filtered_df
    .groupby(["Company", "Sentiment"])
    .size()
    .reset_index(name="Count")
)

fig_sentiment_company = px.bar(
    sentiment_company,
    x="Company",
    y="Count",
    color="Sentiment",
    barmode="group",
    title="Sentiment Distribution by Company"
)

st.plotly_chart(
    fig_sentiment_company,
    use_container_width=True
)

# ==============================
# INDUSTRY ANALYSIS
# ==============================

st.subheader("Industry Analysis")

industry_stats = (
    filtered_df
    .groupby("Industry")
    .agg(
        Reviews=("Industry", "size"),
        Average_Rating=("Rating", "mean")
    )
    .reset_index()
)

fig_industry = px.bar(
    industry_stats,
    x="Industry",
    y="Average_Rating",
    title="Average Rating by Industry",
    text="Average_Rating"
)

fig_industry.update_traces(
    texttemplate="%{text:.2f}"
)

st.plotly_chart(
    fig_industry,
    use_container_width=True
)

# ==============================
# REVIEW LENGTH
# ==============================

filtered_df["Review_Length"] = (
    filtered_df["Review Text"]
    .astype(str)
    .str.len()
)

average_length = filtered_df["Review_Length"].mean()

st.subheader("Review Length Analysis")

st.metric(
    "Average Review Length",
    f"{average_length:.0f} characters"
)

fig_length = px.histogram(
    filtered_df,
    x="Review_Length",
    nbins=40,
    title="Review Length Distribution"
)

st.plotly_chart(
    fig_length,
    use_container_width=True
)

# ==============================
# DATA TABLE
# ==============================

st.subheader("Review Dataset")

st.dataframe(
    filtered_df[
        [
            "Company",
            "Industry",
            "Review Title",
            "Review Text",
            "Rating",
            "Date",
            "Sentiment"
        ]
    ],
    use_container_width=True,
    height=400
)

# ==============================
# FOOTER
# ==============================

st.divider()

st.caption(
    "Trustpilot Sentiment Analysis | "
    "Dataset: 1572 Reviews | "
    "8 Companies"
)