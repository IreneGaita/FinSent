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

def train_word2vec_model():
    print(">>> [TRAIN] Inizio training Word2Vec")

    # Percorsi
    X_path = "/opt/airflow/data/embeddings/embeddings_word2vec.npy"
    csv_path = "/opt/airflow/data/topic/topics_aggressive.csv"
    model_path = "/opt/airflow/data/models/model_word2vec.pkl"
    metrics_path = "/opt/airflow/data/models/metrics_word2vec.json"

    # Caricamento dati
    X = np.load(X_path)
    df = pd.read_csv(csv_path)
    y = df["label"]
    print(f">>> [TRAIN] X shape: {X.shape}, y shape: {y.shape}")

    #  80% training-20% test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classificatore
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metriche (Calcola l’accuracy-Crea un classification report (precision, recall, F1 per ogni classe))
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f">>> [TRAIN] Accuracy: {acc}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Salvataggio modello
    joblib.dump(clf, model_path)
    print(f">>> [TRAIN] Modello salvato in {model_path}")

    # Salvataggio metriche
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)
    print(f">>> [TRAIN] Metriche salvate in {metrics_path}")

with DAG(
    dag_id="train_word2vec_model_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "train", "word2vec"],
) as dag:

    train_model = PythonOperator(
        task_id="train_word2vec_model",
        python_callable=train_word2vec_model,
    )

    train_model
