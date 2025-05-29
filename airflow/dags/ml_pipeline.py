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
from dag_light_preprocessing import (
    light_preprocess_for_bertopic,
    load_light_data_preprocessed
)

with DAG("ml_pipeline",
         start_date=datetime(2023, 1, 1),
         schedule_interval=None,
         catchup=False,
         tags=["mlops"]) as dag:

    # Full pipeline
    task_run_preprocessing = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing,
        op_kwargs= {'input_path': '/opt/airflow/data/raw/all-data.csv', 'pipeline_type': 'full'}
    )
    task_load_data_preprocessed = PythonOperator(
        task_id='load_data_preprocessed',
        python_callable=load_data_preprocessed,
        op_kwargs={'csv_input': '/opt/airflow/data/processed/preprocessed_data.csv','pipeline_type': 'full'}
    )

    task_load_data_light = PythonOperator(
        task_id='load_data_light',
        python_callable=load_light_data_preprocessed,
        op_kwargs={
            'csv_input': '/opt/airflow/data/processed/light_preprocessed_data.csv','pipeline_type': 'light'
        },
    )

    task_generate_embeddings_full = PythonOperator(
        task_id='generate_embeddings_full',
        python_callable=generate_embeddings
    )

    task_test_hdbscan_params_full = PythonOperator(
        task_id='test_hdbscan_params_full',
        python_callable=test_hdbscan_params
    )

    task_run_bertopic_full = PythonOperator(
        task_id='run_bertopic_full',
        python_callable=run_bertopic
    )

    # Light pipeline
    task_light_preprocess = PythonOperator(
        task_id='run_light_preprocessing',
        python_callable=light_preprocess_for_bertopic,
        op_kwargs={
            'csv_input': '/opt/airflow/data/raw/all-data.csv',
            'csv_output': '/opt/airflow/data/processed/light_preprocessed_data.csv'
        }
    )

    task_generate_embeddings_light = PythonOperator(
        task_id='generate_embeddings_light',
        python_callable=generate_embeddings,

    )

    task_test_hdbscan_params_light = PythonOperator(
        task_id='test_hdbscan_params_light',
        python_callable=test_hdbscan_params,

    )

    task_run_bertopic_light = PythonOperator(
        task_id='run_bertopic_light',
        python_callable=run_bertopic,
        op_kwargs={'pipeline_type': 'light'},

    )

    # Set dependencies
    task_run_preprocessing >> task_load_data_preprocessed >> task_generate_embeddings_full >> task_test_hdbscan_params_full >> task_run_bertopic_full
    task_light_preprocess >> task_load_data_light >> task_generate_embeddings_light >> task_test_hdbscan_params_light >> task_run_bertopic_light
