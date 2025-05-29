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

    task_load_data_light = PythonOperator(
        task_id='load_data_light',
        python_callable=load_light_data_preprocessed,
        op_kwargs={
            'csv_input': '/opt/airflow/data/processed/light_preprocessed_data.csv','pipeline_type': 'light'
        },
    )

    for pipeline_type in ['full', 'light']:
        load_task = PythonOperator(
            task_id=f'load_data_{pipeline_type}',
            python_callable=load_data_preprocessed,
            op_kwargs={'csv_input': f'/opt/airflow/data/processed/cleaned_data_{pipeline_type}.csv',
                       'pipeline_type': pipeline_type},
            provide_context=True
        )

        embeddings_task = PythonOperator(
            task_id=f'generate_embeddings_{pipeline_type}',
            python_callable=generate_embeddings,
            op_kwargs={'pipeline_type': pipeline_type},
            provide_context=True
        )

        hdbscan_task = PythonOperator(
            task_id=f'test_hdbscan_params_{pipeline_type}',
            python_callable=test_hdbscan_params,
            op_kwargs={'pipeline_type': pipeline_type},
            provide_context=True
        )

        bertopic_task = PythonOperator(
            task_id=f'run_bertopic_{pipeline_type}',
            python_callable=run_bertopic,
            op_kwargs={'pipeline_type': pipeline_type},
            provide_context=True
        )






    # Set dependencies
    task_run_preprocessing >> load_task >> embeddings_task >> hdbscan_task >> bertopic_task
    load_task >> task_load_data_light >> embeddings_task >> hdbscan_task >> bertopic_task
