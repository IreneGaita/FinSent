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

    # --- Preprocessing ---
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

    # --- BERTopic ---
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

    # --- Embeddings ---
    log_start_embeddings = PythonOperator(
        task_id='log_start_embeddings',
        python_callable=log_message("🔁 Avvio DAG embeddings_generation")
    )

    trigger_embeddings = TriggerDagRunOperator(
        task_id='trigger_embeddings_generation',
        trigger_dag_id='generate_embeddings_dag',
        wait_for_completion=True,
        reset_dag_run=True
    )

    # --- Training ---
    log_start_training_w2v = PythonOperator(
        task_id='log_start_training_word2vec_balanced',
        python_callable=log_message("🔁 Avvio DAG training_word2vec")
    )

    trigger_training_w2v = TriggerDagRunOperator(
        task_id='trigger_training_word2vec_balanced',
        trigger_dag_id='train_word2vec_model_dag',
        wait_for_completion=True,
        reset_dag_run=True
    )

    log_start_training_w2v_unbalanced = PythonOperator(
        task_id='log_start_training_word2vec_unbalanced',
        python_callable=log_message("🔁 Avvio DAG training_word2vec UNBALANCED")
    )

    trigger_training_w2v_unbalanced = TriggerDagRunOperator(
        task_id='trigger_training_word2vec_unbalanced',
        trigger_dag_id='train_word2vec_model_dag_unbalanced',
        wait_for_completion=True,
        reset_dag_run=True
    )

    log_start_training_sbert = PythonOperator(
        task_id='log_start_training_sbert',
        python_callable=log_message("🔁 Avvio DAG training_sbert")
    )

    trigger_training_sbert = TriggerDagRunOperator(
        task_id='trigger_training_sbert',
        trigger_dag_id='train_sbert_model_dag',
        wait_for_completion=True,
        reset_dag_run=True
    )

    log_start_training_sbert_unbalanced = PythonOperator(
        task_id='log_start_training_sbert_unbalanced',
        python_callable=log_message("🔁 Avvio DAG training_sbert UNBALANCED")
    )

    trigger_training_sbert_unbalanced = TriggerDagRunOperator(
        task_id='trigger_training_sbert_unbalanced',
        trigger_dag_id='train_sbert_model_dag_unbalanced',
        wait_for_completion=True,
        reset_dag_run=True
    )

    # --- Confronto ---
    log_start_comparison = PythonOperator(
        task_id='log_start_comparison',
        python_callable=log_message("📊 Avvio DAG di confronto esperimenti")
    )

    trigger_comparison = TriggerDagRunOperator(
        task_id='trigger_compare_experiments',
        trigger_dag_id='compare_experiments_dag',
        wait_for_completion=True,
        reset_dag_run=True
    )

    # --- Best Model DAG ---
    log_start_best_model = PythonOperator(
        task_id='log_start_best_model_update',
        python_callable=log_message("✅ Avvio DAG aggiornamento best model")
    )

    trigger_best_model = TriggerDagRunOperator(
        task_id='trigger_best_model_update',
        trigger_dag_id='compare_and_update_best_model_dag',
        wait_for_completion=True,
        reset_dag_run=True
    )

    # --- Definizione dipendenze ---

    [
        log_start_preprocessing >> trigger_preprocessing,
        log_start_light >> trigger_light
    ] >> log_start_bertopic >> trigger_bertopic \
    >> log_start_embeddings >> trigger_embeddings

    trigger_embeddings >> [
        log_start_training_w2v >> trigger_training_w2v,
        log_start_training_sbert >> trigger_training_sbert,
        log_start_training_w2v_unbalanced >> trigger_training_w2v_unbalanced,
        log_start_training_sbert_unbalanced >> trigger_training_sbert_unbalanced
    ] >> log_start_comparison >> trigger_comparison \
        >> log_start_best_model >> trigger_best_model
