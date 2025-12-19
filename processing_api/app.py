from fastapi import FastAPI, Request
import joblib
import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer

# ------------------- INIT APP -------------------
app = FastAPI(title="AeroStream Processing API")

# ------------------- LOAD MODEL -------------------
MODEL_PATH = os.path.join("modele", "logistic_model_final.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

classifier = joblib.load(MODEL_PATH)

# ------------------- LOAD EMBEDDING MODEL -------------------
embedding_model = SentenceTransformer(
    "cardiffnlp/twitter-roberta-base-sentiment"
)

# ------------------- CLEAN TEXT -------------------
def clean_text(text):
    if text is None:
        return None

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None

# ------------------- PREDICTION ENDPOINT -------------------
@app.post("/predict")
async def predict(request: Request):
    """
    Expected JSON:
    {
        "texts": ["tweet 1", "tweet 2", "..."]
    }
    """

    data = await request.json()
    texts = data.get("texts")

    if not texts or not isinstance(texts, list):
        return {"error": "texts must be a non-empty list"}

    # Convert to DataFrame
    df = pd.DataFrame(texts, columns=["text"])

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove null & duplicates
    df = df.dropna().drop_duplicates(subset="text")

    if df.empty:
        return {"error": "No valid text after cleaning"}

    # ------------------- EMBEDDING -------------------
    embeddings = embedding_model.encode(
        df["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # ------------------- PREDICTION -------------------
    predictions = classifier.predict(embeddings)

    # ------------------- RESPONSE -------------------
    results = []
    for text, pred in zip(df["text"], predictions):
        results.append({
            "text": text,
            "sentiment": int(pred)
        })

    return {
        "count": len(results),
        "results": results
    }
