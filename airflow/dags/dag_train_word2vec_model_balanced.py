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

def train_word2vec_model(use_smote=False):
    print(">>> [TRAIN] Inizio training Word2Vec")

    # Percorsi base
    X_path = "/opt/airflow/data/embeddings/embeddings_word2vec.npy"
    csv_path = "/opt/airflow/data/topic/topics_aggressive.csv"

    # Nome esperimento
    experiment_name = "word2vec_balanced"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    experiment_dir = f"/opt/airflow/data/models/{experiment_name}_{timestamp}"

    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    config_path = os.path.join(experiment_dir, "config.json")

    # Caricamento dati
    X = np.load(X_path)
    df = pd.read_csv(csv_path)
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

    # Salvataggio directory esperimento
    os.makedirs(experiment_dir, exist_ok=True)

    # Salvataggio modello
    joblib.dump(clf, model_path)
    print(f">>> [TRAIN] Modello salvato in {model_path}")

    # Salvataggio metriche
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)
    print(f">>> [TRAIN] Metriche salvate in {metrics_path}")

    # Salvataggio configurazione esperimento
    config = {
        "model": "LogisticRegression",
        "embedding": "Word2Vec",
        "balancing": "class_weight=balanced",
        "use_smote": use_smote,
        "test_size": 0.2,
        "random_state": 42,
        "timestamp": timestamp
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f">>> [TRAIN] Config salvata in {config_path}")

# DAG
with DAG(
    dag_id="train_word2vec_model_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "train", "word2vec"],
) as dag:

    train_model = PythonOperator(
        task_id="train_word2vec_model",
        python_callable=lambda: train_word2vec_model(use_smote=False),
    )

    train_model
