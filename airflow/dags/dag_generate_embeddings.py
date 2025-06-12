from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import torch
from airflow.operators.bash import BashOperator

# Percorsi
DATASET_PATH = "/opt/airflow/data/topic/topics_aggressive.csv"
EMBEDDING_DIR = "opt/airflow/data/embeddings/"
WORD2VEC_PATH = "opt/airflow/data/embeddings/GoogleNews-vectors-negative300.bin"



# Funzioni embedding
def generate_word2vec():
    import pandas as pd
    import numpy as np
    import gensim.downloader as api
    import os
    from tqdm import tqdm

    print(">>> [WORD2VEC] Task avviato")

    # Carica il dataset
    df = pd.read_csv("/opt/airflow/data/topic/topics_aggressive.csv").sample(5, random_state=42)
    print(f">>> [WORD2VEC] Dataset caricato: {df.shape}")

    # Carica il modello GloVe (50 dimensioni, leggero)
    print(">>> [WORD2VEC] Caricamento modello glove-wiki-gigaword-50")
    model = api.load("glove-wiki-gigaword-50")

    embeddings = []
    for text in tqdm(df["cleaned_text"], desc="Word2Vec (GloVe)"):
        words = text.split()
        vectors = [model[w] for w in words if w in model]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))

    output_path = "/opt/airflow/data/embeddings"
    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}/embeddings_word2vec.npy", np.array(embeddings))

    print(">>> [WORD2VEC] Embedding salvati")



def generate_sbert():
    import pandas as pd
    import numpy as np
    import os
    from sentence_transformers import SentenceTransformer

    print(">>> [SBERT] Inizio task SBERT")

    # Percorsi espliciti
    dataset_path = "/opt/airflow/data/topic/topics_aggressive.csv"
    output_path = "/opt/airflow/data/embeddings/embeddings_sbert.npy"

    # Carica dati
    df = pd.read_csv(dataset_path)
    print(f">>> [SBERT] Dataset caricato con shape: {df.shape}")

    # Carica modello e genera embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["cleaned_text"], show_progress_bar=True)
    print(">>> [SBERT] Embedding generati")

    # Salva
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f">>> [SBERT] Embedding salvati in {output_path}")



# DAG

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 0
}

with DAG(
        dag_id="generate_embeddings_dag",
        default_args=default_args,
        schedule_interval=None,
        catchup=False,
        description="Genera Word2Vec, SBERT, BERT, FinBERT embeddings"
) as dag:
    task_word2vec = PythonOperator(
        task_id="generate_word2vec_embeddings",
        python_callable=generate_word2vec
    )

    task_sbert = PythonOperator(
        task_id="generate_sbert_embeddings",
        python_callable=generate_sbert
    )



task_word2vec>>task_sbert
