from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer

MODEL_PATH = "/opt/airflow/data/models/best_model_latest/model.pkl"

# Load model and SBERT
try:
    print("🔁 Loading model...")
    model = joblib.load(MODEL_PATH)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model and SBERT loaded.")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

app = FastAPI(title="Financial Sentiment Inference API")

class TextInput(BaseModel):
    texts: list[str]

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": str(datetime.now())}

@app.post("/predict")
def predict(input_data: TextInput):
    try:
        embeddings = sbert.encode(input_data.texts)
        predictions = model.predict(embeddings)
        return {
            "predictions": predictions.tolist(),
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
