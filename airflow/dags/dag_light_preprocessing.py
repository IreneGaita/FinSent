from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import re
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def light_preprocess_for_bertopic(csv_input: str, csv_output: str):
    logger.info("=== INIZIO LIGHT PREPROCESSAMENTO PER BERTopic ===")
    try:
        df = pd.read_csv(csv_input, encoding='latin1', names=['label', 'text'])
        logger.info(f"✓ File caricato: {csv_input} | Righe: {df.shape[0]}")
    except Exception as e:
        logger.error(f"✗ Errore nel caricamento del CSV: {str(e)}")
        raise

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[\$€£]', ' CUR ', text)
        text = re.sub(r'\b\d{1,3}(?:[.,]\d+)?\s*%', ' NUM_PERCENT ', text)
        text = re.sub(r'\b\d{1,3}(?:[.,]?\d{3})*(?:[.,]\d+)?\b', ' NUM ', text)
        return text

    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.split().str.len() > 2]

    df.to_csv(csv_output, index=False)
    logger.info(f"✓ File preprocessato salvato in: {csv_output}")

def load_light_data_preprocessed(csv_input: str, **context):
    try:
        assert os.path.exists(csv_input), f"File non trovato: {csv_input}"
        df = pd.read_csv(csv_input)
        if 'cleaned_text' not in df.columns:
            raise ValueError("Colonna 'cleaned_text' mancante nel CSV")
        logger.info(f"[INFO] Dataset caricato correttamente: {csv_input} con {df.shape[0]} righe")
        # Passa dataframe come XCom (attenzione dimensione)
        context['ti'].xcom_push(key='dataframe_light', value=df)

    except Exception as e:
        logger.error(f"[ERROR] Errore nel caricamento del dataset: {e}")
        raise

with DAG(
        dag_id='light_preprocessing',
        start_date=datetime(2023, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=['mlops', 'preprocessing', 'bertopic'],
) as dag:

    task_light_preprocess = PythonOperator(
        task_id='run_light_preprocessing',
        python_callable=light_preprocess_for_bertopic,
        op_kwargs={
            'csv_input': '/opt/airflow/data/raw/all-data.csv',
            'csv_output': '/opt/airflow/data/processed/light_preprocessed_data.csv'
        }
    )

    task_load_light_data = PythonOperator(
        task_id='load_light_data_preprocessed',
        python_callable=load_light_data_preprocessed,
        op_kwargs={
            'csv_input': '/opt/airflow/data/processed/light_preprocessed_data.csv'
        },
        provide_context=True,
    )

    task_light_preprocess >> task_load_light_data
