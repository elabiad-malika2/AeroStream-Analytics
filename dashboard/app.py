import streamlit as st
import json
import pandas as pd
from sqlalchemy import (
    create_engine, Table, Column, Integer, String, MetaData,
    select, func
)
from streamlit_autorefresh import st_autorefresh
from PIL import Image


# =============================
# CONFIG PAGE
# =============================
st.set_page_config(
    page_title="AeroStream Analytics",
    layout="wide"
)

# Refresh every 10 seconds
st_autorefresh(interval=10000, key="refresh")

page = st.sidebar.radio(
    "Navigation",
    ["Aggregation Dashboard", "Evaluation des métriques"]
)

# =============================
# DATABASE CONNECTION
# =============================
DB_URL = "postgresql+psycopg2://postgres:malika123@postgres:5432/aerostream"
engine = create_engine(DB_URL)

metadata = MetaData()

tweets = Table(
    "tweets",
    metadata,
    Column("id", Integer),
    Column("text", String),
    Column("airline", String),
    Column("sentiment", String),
    Column("negativereason", String),
)

if page == "Aggregation Dashboard":
    # =============================
    # KPI FUNCTIONS
    # =============================
    @st.cache_data(ttl=8)
    def total_tweets():
        stmt = select(func.count()).select_from(tweets)
        with engine.connect() as conn:
            return conn.execute(stmt).fetchone()[0]


    @st.cache_data(ttl=8)
    def airlines_count():
        stmt = select(func.count(func.distinct(tweets.c.airline)))
        with engine.connect() as conn:
            return conn.execute(stmt).fetchone()[0]


    @st.cache_data(ttl=8)
    def negative_percentage():
        stmt_total = select(func.count()).select_from(tweets)
        stmt_negative = select(func.count()).where(tweets.c.sentiment == "negative")

        with engine.connect() as conn:
            total = conn.execute(stmt_total).fetchone()[0]
            negative = conn.execute(stmt_negative).fetchone()[0]

        if total == 0:
            return 0

        return round((negative / total) * 100, 2)

    # =============================
    # AGGREGATIONS
    # =============================
    @st.cache_data(ttl=8)
    def tweets_per_airline():
        stmt = (
            select(
                tweets.c.airline,
                func.count().label("tweets")
            )
            .group_by(tweets.c.airline)
        )

        with engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        return pd.DataFrame(rows, columns=["airline", "tweets"])


    @st.cache_data(ttl=8)
    def sentiment_by_airline():
        stmt = (
            select(
                tweets.c.airline,
                tweets.c.sentiment,
                func.count().label("count")
            )
            .group_by(tweets.c.airline, tweets.c.sentiment)
        )

        with engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        return pd.DataFrame(rows, columns=["airline", "sentiment", "count"])


    @st.cache_data(ttl=8)
    def satisfaction_rate():
        stmt = (
            select(
                tweets.c.airline,
                tweets.c.sentiment,
                func.count().label("count")
            )
            .group_by(tweets.c.airline, tweets.c.sentiment)
        )

        with engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        df = pd.DataFrame(rows, columns=["airline", "sentiment", "count"])

        result = []

        for airline in df["airline"].unique():
            airline_df = df[df["airline"] == airline]
            total = airline_df["count"].sum()
            positive = airline_df[airline_df["sentiment"] == "positive"]["count"].sum()
            rate = round((positive / total) * 100, 2) if total > 0 else 0
            result.append([airline, rate])

        return pd.DataFrame(result, columns=["airline", "satisfaction_rate"])


    @st.cache_data(ttl=8)
    def negative_reasons():
        stmt = (
            select(
                tweets.c.negativereason,
                func.count().label("count")
            )
            .where(tweets.c.sentiment == "negative")
            .group_by(tweets.c.negativereason)
        )

        with engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        return pd.DataFrame(rows, columns=["negativereason", "count"])

    # =============================
    # UI
    # =============================
    st.title(" AeroStream – Real-time Sentiment Dashboard")
    st.markdown("Analyse en temps réel des avis clients des compagnies aériennes")

    # =============================
    # KPI CARDS
    # =============================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Tweets", total_tweets())
    col2.metric("Compagnies aériennes", airlines_count())
    col3.metric("Tweets négatifs (%)", f"{negative_percentage()}%")

    st.divider()

    # =============================
    # VISUALISATIONS
    # =============================
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(" Volume de tweets par compagnie")
        df_airline = tweets_per_airline()
        st.bar_chart(df_airline.set_index("airline"))

    with col_right:
        st.subheader(" Répartition des sentiments par compagnie")

        df_sentiment = sentiment_by_airline()

        pivot_sentiment = df_sentiment.pivot(
            index="airline",
            columns="sentiment",
            values="count"
        ).fillna(0)

        st.bar_chart(pivot_sentiment)

    st.divider()

    # =============================
    # SATISFACTION
    # =============================
    st.subheader(" Taux de satisfaction par compagnie")

    df_satisfaction = satisfaction_rate()
    st.bar_chart(df_satisfaction.set_index("airline"))

    st.divider()

    # =============================
    # NEGATIVE REASONS
    # =============================
    st.subheader(" Principales causes de tweets négatifs")

    neg_df = negative_reasons()
    if not neg_df.empty:
        st.bar_chart(neg_df.set_index("negativereason"))
    else:
        st.info("Aucune donnée négative pour le moment")


elif page == "Evaluation des métriques":


    st.title("Évaluation des Modèles ML")

    # =============================
    # LOAD METRICS JSON
    # =============================
    with open("Evaluation/metrics_results.json") as f:
        metrics = json.load(f)


    # =============================
    # PER MODEL DETAILS
    # =============================

    for model, data in metrics.items():
        st.subheader(f" {model}")

        df_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "F1 weighted", "Gap Accuracy"],
            "Value": [
                data["test"]["accuracy"],
                data["test"]["f1_weighted"],
                data["gap"]["accuracy"]
            ]
        }).set_index("Metric")

        st.bar_chart(df_metrics)


    st.subheader("Matrice de confusion – Logistic Regression")
    st.image(
        Image.open("Evaluation/confusion_matrix_logistic.png"),
        use_column_width=True
    )

   
    st.subheader("Comparaison ROC AUC")
    st.image(
        Image.open("Evaluation/roc_auc_comparison.png"),
        use_column_width=True
    )

    st.divider()


    st.subheader("Interprétation")
    st.markdown("""
    - **Logistic Regression** présente le meilleur compromis biais / variance.
    - L’écart train/test est faible → bonne généralisation.
    - Le modèle **SVM** montre un sur-apprentissage plus marqué.
    - Les courbes ROC-AUC confirment une bonne séparation multi-classes.
    """)
