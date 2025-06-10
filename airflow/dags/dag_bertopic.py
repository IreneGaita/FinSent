from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from bertopic import BERTopic
import logging
from sentence_transformers import SentenceTransformer


# ================================
# Funzione con logging dettagliato
# ================================


def apply_bertopic(csv_path: str, output_path: str, text_column: str = "cleaned_text", **kwargs):
    logger = logging.getLogger("airflow.task")
    logger.info("Inizio funzione apply_bertopic")
    logger.info(f"Lettura file CSV da: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"CSV letto correttamente: {len(df)} righe")
    documents = df[text_column].dropna().tolist()
    logger.info(f"Numero di documenti da processare: {len(documents)}")

    # Embedding model esplicitamente separato
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(embedding_model=embedding_model)

    try:
        topics, _ = topic_model.fit_transform(documents)
        logger.info("BERTopic applicato correttamente")
    except Exception as e:
        logger.error(f"Errore durante il fit_transform: {e}")
        raise

    df['topic'] = topics
    topic_labels = {
        topic: topic_model.get_topic(topic)[0][0]
        for topic in set(topics) if topic != -1
    }
    df['topic_name'] = df['topic'].map(topic_labels).fillna("Outlier")

    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Risultati salvati in: {output_path}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio del file CSV: {e}")
        raise


# ====================
# Parametri del DAG
# ====================
default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
}

dag = DAG(
    'bertopic_topic_modeling',
    default_args=default_args,
    description='Applica BERTopic a due dataset preprocessati con logging',
    schedule_interval=None,
    catchup=False
)

# ====================
# Task: BERTopic Aggressive
# ====================
task_aggressive = PythonOperator(
    task_id='bertopic_on_aggressive',
    python_callable=apply_bertopic,
    op_kwargs={
        'csv_path': '/opt/airflow/data/processed/preprocessed_data.csv',
        'output_path': '/opt/airflow/data/topic/topics_aggressive.csv',
        'text_column': 'text'
    },
    dag=dag
)

# ====================
# Task: BERTopic Light
# ====================
task_light = PythonOperator(
    task_id='bertopic_on_light',
    python_callable=apply_bertopic,
    op_kwargs={
        'csv_path': '/opt/airflow/data/processed/light_preprocessed_data.csv',
        'output_path': '/opt/airflow/data/topic/topics_light.csv',
        'text_column': 'text'
    },
    dag=dag
)

task_aggressive >> task_light