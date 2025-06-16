from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json
import pandas as pd

def compare_experiments():
    models_dir = "/opt/airflow/data/models"
    output_path = "/opt/airflow/data/reports/comparison_table.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    for exp_dir in os.listdir(models_dir):
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
            "f1_neutral": metrics["report"].get("neutral", {}).get("f1-score"),
            "f1_positive": metrics["report"].get("positive", {}).get("f1-score"),
            "f1_negative": metrics["report"].get("negative", {}).get("f1-score"),
            "timestamp": config.get("timestamp")
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="accuracy", ascending=False)
    df.to_csv(output_path, index=False)
    print(f">>> 📊 Comparazione salvata in: {output_path}")

# DAG
with DAG(
    dag_id="compare_experiments_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "evaluation", "compare"],
) as dag:

    run_comparison = PythonOperator(
        task_id="compare_experiments",
        python_callable=compare_experiments,
    )

    run_comparison
