from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG('main_nlp_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    # Trigger DAG di preprocessing (aggressivo)
    trigger_preprocessing = TriggerDagRunOperator(
        task_id='trigger_preprocessing',
        trigger_dag_id='preprocessing_pipeline',
        wait_for_completion=True,         # Aspetta che finisca
        poke_interval=30,                 # Intervallo di polling
        reset_dag_run=True                # Riavvia il DAG se già esiste un run
    )

    # Trigger DAG di preprocessing (leggero)
    trigger_light = TriggerDagRunOperator(
        task_id='trigger_light_preprocessing',
        trigger_dag_id='bertopic_light_preprocessing_pipeline',
        wait_for_completion=True,
        poke_interval=30,
        reset_dag_run=True
    )

    # Una volta finiti entrambi, triggera il DAG di BERTopic
    trigger_bertopic = TriggerDagRunOperator(
        task_id='trigger_bertopic',
        trigger_dag_id='bertopic_topic_modeling',
        wait_for_completion=False,  # opzionale, metti True se vuoi aspettare anche questo
        reset_dag_run=True
    )

    # Collegamento: avvia entrambi i preprocessing, poi bertopic
    [trigger_preprocessing, trigger_light] >> trigger_bertopic