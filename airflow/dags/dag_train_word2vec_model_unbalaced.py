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

def train_word2vec_model_unbalanced():
    print(">>> [TRAIN] Inizio training Word2Vec senza bilanciamento classi")

    # Percorsi base
    X_path = "/opt/airflow/data/embeddings/embeddings_word2vec.npy"
    csv_path = "/opt/airflow/data/topic/topics_aggressive.csv"

    # Nome esperimento
    experiment_name = "word2vec_unbalanced"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # Cartelle storiche (timestampate)
    experiment_dir = f"/opt/airflow/data/models/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    config_path = os.path.join(experiment_dir, "config.json")

    # Cartella latest
    latest_dir = "/opt/airflow/data/models/word2vec_unbalanced_latest"
    os.makedirs(latest_dir, exist_ok=True)
    latest_model = os.path.join(latest_dir, "model.pkl")
    latest_metrics = os.path.join(latest_dir, "metrics.json")
    latest_config = os.path.join(latest_dir, "config.json")

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

    # Classificatore base (no bilanciamento)
    clf = LogisticRegression(
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

    # Salvataggio metriche
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)
    with open(latest_metrics, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)
    print(f">>> [TRAIN] Metriche salvate in {metrics_path} e {latest_metrics}")

    # Salvataggio configurazione esperimento
    config = {
        "model": "LogisticRegression",
        "embedding": "Word2Vec",
        "balancing": "none",
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
with DAG(
    dag_id="train_word2vec_model_dag_unbalanced",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "train", "word2vec", "unbalanced"],
) as dag:

    train_model = PythonOperator(
        task_id="train_word2vec_model_unbalanced",
        python_callable=train_word2vec_model_unbalanced,
    )

    train_model
