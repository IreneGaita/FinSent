from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import logging
import os

# CONFIGURAZIONE LOGGING
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#CONFIGURAZIONE NLTK
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)


def download_nltk_resources():
    logger.info("Download risorse NLTK...")
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"✓ '{resource}' scaricato.")
        except Exception as e:
            logger.error(f"✗ Errore download '{resource}': {str(e)}")
            raise


def preprocess():
    logger.info("=== INIZIO PREPROCESSAMENTO ===")
    download_nltk_resources()

    csv_input = "/opt/airflow/dataset/sentiment_data/raw/all-data.csv"
    csv_output = "/opt/airflow/dataset/sentiment_data/processed/preprocessed_data.csv"

    try:
        df = pd.read_csv(csv_input, encoding='latin1', names=['label', 'text'])
        logger.info(f"Caricato file: {csv_input} | Righe: {df.shape[0]}")
    except Exception as e:
        logger.error(f"Errore caricamento CSV: {str(e)}")
        raise

    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\d+', '', text.lower())
        text = re.sub(r'[^\w\s]', '', text)

        try:
            nltk.data.find('tokenizers/punkt')
            words = nltk.word_tokenize(text)
        except Exception:
            words = text.split()  # Fallback

        words = [stemmer.stem(w) for w in words if w not in stop_words]
        return ' '.join([lemmatizer.lemmatize(w) for w in words])

    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df = df[df['cleaned_text'].str.split().str.len() > 2]

    try:
        df.to_csv(csv_output, index=False)
        logger.info(f"✓ Preprocessamento completato. File salvato: {csv_output}Righe: {df.shape[0]}")
    except Exception as e:
        logger.error(f"Errore salvataggio file: {str(e)}")
        raise


def run_preprocessing():
    try:
        preprocess()
        logger.info("✓ Task completato con successo.")
    except Exception as e:
        logger.error(f"✗ Errore nel task di preprocessing: {str(e)}")


with DAG(
    dag_id='preprocessing_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'preprocessing'],
) as dag:
    task_preprocess = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing,
    )
