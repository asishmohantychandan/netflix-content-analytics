
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Netflix Content Analytics",
    page_icon="🎬",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# 2. LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        ROOT / "data" / "netflix_cleaned.csv",
        parse_dates=["date_added"]
    )

    recommendation_data = pd.read_csv(
        ROOT / "data" / "recommendation_data.csv"
    )

    forecast_path = ROOT / "data" / "netflix_monthly_forecast.csv"
    forecast = pd.read_csv(forecast_path)

    forecast_comparison = pd.read_csv(
        ROOT / "reports" / "forecast_model_comparison.csv"
    )

    classification_comparison = pd.read_csv(
        ROOT / "reports" / "classification_model_comparison.csv"
    )

    return (
        df,
        recommendation_data,
        forecast,
        forecast_comparison,
        classification_comparison
    )


@st.cache_resource
def load_models():
    vectorizer = joblib.load(
        ROOT / "models" / "tfidf_vectorizer.pkl"
    )

    tfidf_matrix = load_npz(
        ROOT / "models" / "tfidf_matrix.npz"
    )

    classifier = joblib.load(
        ROOT / "models" / "netflix_type_classifier.pkl"
    )

    return vectorizer, tfidf_matrix, classifier


df, recommendation_data, forecast, forecast_comparison, classification_comparison = load_data()

vectorizer, tfidf_matrix, classifier = load_models()


# --------------------------------------------------
# 3. TITLE AND SIDEBAR FILTERS
# --------------------------------------------------

st.title("🎬 Netflix Content Analytics Dashboard")

st.markdown(
    "Explore Netflix's content catalogue, trends, "
    "recommendations and machine learning insights."
)

st.sidebar.header("Dashboard Filters")

content_types = st.sidebar.multiselect(
    "Content Type",
    options=sorted(df["type"].dropna().unique()),
    default=sorted(df["type"].dropna().unique())
)

years = sorted(df["release_year"].dropna().unique())

selected_years = st.sidebar.slider(
    "Release Year Range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

filtered_df = df[
    (df["type"].isin(content_types)) &
    (df["release_year"].between(*selected_years))
].copy()


# --------------------------------------------------
# 4. OVERVIEW KPIs
# --------------------------------------------------

st.subheader("Catalogue Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", f"{len(filtered_df):,}")

col2.metric(
    "Movies",
    f"{(filtered_df['type'] == 'Movie').sum():,}"
)

col3.metric(
    "TV Shows",
    f"{(filtered_df['type'] == 'TV Show').sum():,}"
)

col4.metric(
    "Countries",
    f"{filtered_df['country'].nunique():,}"
)

st.divider()


# --------------------------------------------------
# 5. CONTENT ANALYSIS
# --------------------------------------------------

st.subheader("Content Distribution")

col1, col2 = st.columns(2)

with col1:
    type_counts = filtered_df["type"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        type_counts,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Movies vs TV Shows")
    st.pyplot(fig)
    plt.close(fig)


with col2:
    yearly_counts = (
        filtered_df["date_added_year"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots()
    yearly_counts.plot(kind="bar", ax=ax)
    ax.set_title("Content Added by Year")
    ax.set_xlabel("Year Added")
    ax.set_ylabel("Number of Titles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 6. TOP COUNTRIES AND GENRES
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Content-Producing Countries")

    countries = (
        filtered_df["country"]
        .dropna()
        .str.split(", ")
        .explode()
    )

    top_countries = countries.value_counts().head(10)

    st.bar_chart(top_countries)


with col2:
    st.subheader("Top 10 Genres")

    genres = (
        filtered_df["listed_in"]
        .dropna()
        .str.split(", ")
        .explode()
    )

    top_genres = genres.value_counts().head(10)

    st.bar_chart(top_genres)


st.divider()


# --------------------------------------------------
# 7. CONTENT TREND FORECAST
# --------------------------------------------------

st.subheader("Monthly Content Addition Forecast")

st.caption(
    "Exploratory forecast based on historical catalogue additions. "
    "The dataset ends in September 2021."
)

st.dataframe(forecast, use_container_width=True)

st.line_chart(
    forecast.set_index(forecast.columns[0])
)


# --------------------------------------------------
# 8. CONTENT RECOMMENDATION SYSTEM
# --------------------------------------------------

st.divider()

st.subheader("🎯 Content Recommendation System")

title_list = sorted(
    recommendation_data["title"].dropna().unique()
)

selected_title = st.selectbox(
    "Choose a Netflix title",
    title_list
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


def get_recommendations(title, top_n=5):

    matches = recommendation_data[
        recommendation_data["title"].str.casefold()
        == title.strip().casefold()
    ]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    selected_type = recommendation_data.loc[idx, "type"]

    scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    candidates = recommendation_data.index[
        (recommendation_data["type"] == selected_type) &
        (recommendation_data.index != idx)
    ]

    ranked = sorted(
        candidates,
        key=lambda i: scores[i],
        reverse=True
    )[:top_n]

    results = recommendation_data.loc[
        ranked,
        ["title", "type", "listed_in"]
    ].copy()

    results["Similarity Score"] = np.round(
        scores[ranked], 3
    )

    return results.reset_index(drop=True)


if st.button("Get Recommendations"):

    recommendations = get_recommendations(
        selected_title,
        top_n
    )

    st.write(f"Recommendations similar to **{selected_title}**")

    st.dataframe(
        recommendations,
        use_container_width=True
    )


# --------------------------------------------------
# 9. MACHINE LEARNING INSIGHTS
# --------------------------------------------------

st.divider()

st.subheader("Machine Learning Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.write("**Classification Model Comparison**")

    st.dataframe(
        classification_comparison.round(4),
        use_container_width=True
    )

with col2:
    st.write("**Forecasting Model Comparison**")

    st.dataframe(
        forecast_comparison.round(3),
        use_container_width=True
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Netflix Content Analytics | Data Science Internship Project"
)
