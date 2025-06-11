from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def log_message(msg):
    def _log(**kwargs):
        logging.info(msg)
    return _log

with DAG('main_nlp_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    log_start_preprocessing = PythonOperator(
        task_id='log_start_preprocessing',
        python_callable=log_message("🔁 Avvio DAG aggressive_preprocessing")
    )

    trigger_preprocessing = TriggerDagRunOperator(
        task_id='trigger_preprocessing',
        trigger_dag_id='aggressive_preprocessing',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True
    )

    log_start_light = PythonOperator(
        task_id='log_start_light',
        python_callable=log_message("🔁 Avvio DAG light_preprocessing")
    )

    trigger_light = TriggerDagRunOperator(
        task_id='trigger_light_preprocessing',
        trigger_dag_id='light_preprocessing',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True
    )

    log_start_bertopic = PythonOperator(
        task_id='log_start_bertopic',
        python_callable=log_message("🔁 Avvio DAG bertopic_topic_modeling")
    )

    trigger_bertopic = TriggerDagRunOperator(
        task_id='trigger_bertopic',
        trigger_dag_id='bertopic_topic_modeling',
        wait_for_completion=True,
        reset_dag_run=True
    )

    [log_start_preprocessing >> trigger_preprocessing,
     log_start_light >> trigger_light] >> log_start_bertopic >> trigger_bertopic
