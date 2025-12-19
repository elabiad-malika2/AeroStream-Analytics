from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from sqlalchemy import create_engine,MetaData,Table,Column,Integer,String,TIMESTAMP
from sqlalchemy.dialects.postgresql import insert

# ------------------- CONFIG -------------------

PRODUCER_API_URL = "http://producer_api:8001/batch"
PROCESSING_API_URL = "http://processing_api:8002/predict"

POSTGRES_CONFIG = "postgresql+psycopg2://postgres:malika123@postgres:5432/aerostream"

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

DEFAULT_ARGS = {
    "owner": "aerostream",
    "retries": 4,
    "retry_delay": timedelta(seconds=30),
}

# ------------------- FUNCTIONS -------------------

def fetch_tweets(**context):
    """Récupération micro-batch depuis l’API du prof"""

    response = requests.get(
        PRODUCER_API_URL,
        params={"batch_size": 20},
        timeout=10
    )
    response.raise_for_status()

    tweets = response.json()
    context["ti"].xcom_push(key="tweets", value=tweets)


def process_tweets(**context):
    tweets = context["ti"].xcom_pull(key="tweets")
    texts = [t["text"] for t in tweets]

    try:
        response = requests.post(
            PROCESSING_API_URL,
            json={"texts": texts},
            timeout=20
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Erreur connexion processing_api:", e)
        raise

    predictions = response.json()["results"]

    processed = []
    for tweet, pred in zip(tweets, predictions):
        processed.append({
            "airline": tweet["airline"],
            "text": pred["text"],
            "sentiment": SENTIMENT_MAP.get(pred["sentiment"], "unknown"),
            "negativereason": tweet["negativereason"],
            "tweet_created": tweet["tweet_created"]
        })

    context["ti"].xcom_push(key="processed", value=processed)



def store_to_postgres(**context):
    """ Stockage brut """

    data = context["ti"].xcom_pull(key="processed")

    if not data:
        return
    
    engine=create_engine(POSTGRES_CONFIG)
    metadata=MetaData()

    # Defintion des tables:

    tweets=Table(
        "tweets",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("airline", String),
        Column("text", String),
        Column("sentiment", String),
        Column("negativereason", String),
        Column("tweet_created", TIMESTAMP),
    )

    # create table if not exist
    metadata.create_all(engine)

    # insertion data

    rows=[
        {
            "airline": row["airline"],
            "text": row["text"],
            "sentiment": row["sentiment"],
            "negativereason": row["negativereason"],
            "tweet_created": row["tweet_created"]
        }
        for row in data
    ]

    with engine.begin() as conn:
        conn.execute(tweets.insert(),rows)


# ------------------- DAG -------------------

with DAG(
    dag_id="aerostream_streaming_pipeline",
    description="Micro-batch streaming ingestion pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/1 * * * *",  # chaque minute
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["streaming", "airflow", "nlp"]
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_tweets",
        python_callable=fetch_tweets
    )

    process_task = PythonOperator(
        task_id="process_tweets",
        python_callable=process_tweets
    )

    store_task = PythonOperator(
        task_id="store_to_postgres",
        python_callable=store_to_postgres
    )

    fetch_task >> process_task >> store_task
