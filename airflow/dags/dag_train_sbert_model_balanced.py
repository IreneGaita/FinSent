from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_sbert_model():
    print(">>> [TRAIN] Inizio training SBERT + Logistic Regression")

    # Percorsi base
    X_path = "/opt/airflow/data/embeddings/embeddings_sbert.npy"
    csv_path = "/opt/airflow/data/topic/topics_aggressive.csv"

    # Nome esperimento
    experiment_name = "sbert_balanced"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # Percorsi storici (timestampati)
    experiment_dir = f"/opt/airflow/data/models/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    config_path = os.path.join(experiment_dir, "config.json")
    vectorizer_path = os.path.join(experiment_dir, "vectorizer.pkl")

    # Percorsi latest (sovrascritti ogni volta)
    latest_dir = "/opt/airflow/data/models/sbert_balanced_latest"
    os.makedirs(latest_dir, exist_ok=True)
    latest_model = os.path.join(latest_dir, "model.pkl")
    latest_metrics = os.path.join(latest_dir, "metrics.json")
    latest_config = os.path.join(latest_dir, "config.json")
    latest_vectorizer = os.path.join(latest_dir, "vectorizer.pkl")

    # Caricamento dati
    X = np.load(X_path)
    df = pd.read_csv(csv_path)

    if len(X) != len(df):
        print(f">>> ⚠️ WARNING: X e y non allineati → X: {len(X)}, y: {len(df)}")
        min_len = min(len(X), len(df))
        X = X[:min_len]
        df = df.iloc[:min_len]

    y = df["label"]
    print(f">>> [TRAIN] X shape: {X.shape}, y shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classificatore con bilanciamento
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metriche
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f">>> [TRAIN] Accuracy: {acc}")

    # Salvataggio modello
    joblib.dump(clf, model_path)
    joblib.dump(clf, latest_model)
    print(f">>> [TRAIN] Modello salvato in {model_path} e {latest_model}")

    # Salvataggio vectorizer SBERT
    print(">>> [TRAIN] Vectorizer SBERT NON salvato: sarà ricaricato dinamicamente")


    # Salvataggio metriche
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)
    with open(latest_metrics, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)
    print(f">>> [TRAIN] Metriche salvate in {metrics_path} e {latest_metrics}")

    # Salvataggio configurazione esperimento
    config = {
        "model": "LogisticRegression",
        "embedding": "SBERT (all-MiniLM-L6-v2)",
        "balancing": "class_weight=balanced",
        "use_smote": False,
        "test_size": 0.2,
        "random_state": 42,
        "timestamp": timestamp
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(latest_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f">>> [TRAIN] Config salvata in {config_path} e {latest_config}")

# DAG
# DAG
with DAG(
    dag_id="train_sbert_model_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "train", "sbert"],
) as dag:

    train_model = PythonOperator(
        task_id="train_sbert_model",
        python_callable=train_sbert_model,
    )

    train_model
