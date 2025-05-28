from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from dag_preprocessing import run_preprocessing
from dag_topic import (
    load_data_preprocessed,
    generate_embeddings,
    test_hdbscan_params,
    run_bertopic
)

with DAG("ml_pipeline",
         start_date=datetime(2023, 1, 1),
         schedule_interval=None,
         catchup=False,
         tags=["mlops"]) as dag:

    t1 = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing
    )

    t2 = PythonOperator(
        task_id='load_data_preprocessed',
        python_callable=load_data_preprocessed
    )

    t3 = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings
    )

    t4 = PythonOperator(
        task_id='test_hdbscan_params',
        python_callable=test_hdbscan_params
    )

    t5 = PythonOperator(
        task_id='run_bertopic',
        python_callable=run_bertopic
    )

    # Dipendenze tra i task
    t1 >> t2 >> t3 >> t4 >> t5