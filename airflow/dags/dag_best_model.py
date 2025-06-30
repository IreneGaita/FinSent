

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json
import pandas as pd
import shutil

# === Funzione 1: confronto tra esperimenti ===
def compare_experiments():
    models_dir = "/opt/airflow/data/models"
    output_path = "/opt/airflow/data/reports/comparison_table.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    for exp_dir in os.listdir(models_dir):
        if "latest" in exp_dir.lower():
            continue  # ignora cartelle latest

        full_path = os.path.join(models_dir, exp_dir)
        if not os.path.isdir(full_path):
            continue

        config_path = os.path.join(full_path, "config.json")
        metrics_path = os.path.join(full_path, "metrics.json")

        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        records.append({
            "experiment": exp_dir,
            "embedding": config.get("embedding"),
            "balancing": config.get("balancing"),
            "accuracy": metrics.get("accuracy"),
            "f1_neutral": metrics.get("report", {}).get("neutral", {}).get("f1-score"),
            "f1_positive": metrics.get("report", {}).get("positive", {}).get("f1-score"),
            "f1_negative": metrics.get("report", {}).get("negative", {}).get("f1-score"),
            "timestamp": config.get("timestamp")
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="accuracy", ascending=False)
    df.to_csv(output_path, index=False)
    print(f">>> 📊 Comparazione salvata in: {output_path}")

# === Funzione 2: aggiorna il modello migliore ===
def update_best_model_latest():
    comparison_path = "/opt/airflow/data/reports/comparison_table.csv"
    models_dir = "/opt/airflow/data/models"
    latest_dir = os.path.join(models_dir, "best_model_latest")

    if not os.path.exists(comparison_path):
        raise FileNotFoundError("Il file comparison_table.csv non esiste.")

    df = pd.read_csv(comparison_path)
    if df.empty:
        raise ValueError("Nessun esperimento trovato per l'aggiornamento.")

    best_row = df.sort_values(by="accuracy", ascending=False).iloc[0]
    best_model_path = os.path.join(models_dir, best_row["experiment"])

    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    os.makedirs(latest_dir)

    for filename in ["model.pkl", "metrics.json", "config.json"]:
        src = os.path.join(best_model_path, filename)
        dst = os.path.join(latest_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f">>> ✅ Modello aggiornato in: {latest_dir}")

# === DAG ===
with DAG(
    dag_id="compare_and_update_best_model_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "evaluation", "update"],
) as dag:

    run_comparison = PythonOperator(
        task_id="compare_experiments",
        python_callable=compare_experiments,
    )

    update_latest_model = PythonOperator(
        task_id="update_best_model_latest",
        python_callable=update_best_model_latest,
    )

    run_comparison >> update_latest_model
