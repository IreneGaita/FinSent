from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Percorsi
DATASET_PATH = "/opt/airflow/data/topic/topics_aggressive.csv"
EMBEDDING_PATH = "/opt/airflow/data/embeddings/embeddings_sbert.npy"
MODEL_DIR = "/opt/airflow/data/models"

def train_sbert_model():
    print(">>> [TRAIN] Inizio training SBERT")

    # Carica dati
    df = pd.read_csv(DATASET_PATH)
    y = df["label"]
    X = np.load(EMBEDDING_PATH)

    print(f">>> [TRAIN] X shape: {X.shape}, y shape: {y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modello
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f">>> [TRAIN] Accuracy: {acc}")

    # Salva output
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model_sbert.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics_sbert.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    print(f">>> [TRAIN] Modello salvato in {model_path}")
    print(f">>> [TRAIN] Metriche salvate in {metrics_path}")

# DAG
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 0
}

with DAG(
    dag_id="train_sbert_model_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Training classificatore su embedding SBERT"
) as dag:
    train_task = PythonOperator(
        task_id="train_sbert_model",
        python_callable=train_sbert_model
    )
