from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import logging
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import dill


# Default args for DAG
default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Define DAG
with DAG('bertopic_topic_modeling',
         schedule_interval=None,
         default_args=default_args,
         catchup=False) as dag:

    def load_data_preprocessed(**context):
        import os
        csv_input = "/opt/airflow/data/processed/preprocessed_data.csv"
        try:
            assert os.path.exists(csv_input), f"File non trovato: {csv_input}"
            df = pd.read_csv(csv_input)
            if 'cleaned_text' not in df.columns:
                raise ValueError("Colonna 'cleaned_text' mancante nel CSV")
            context['ti'].xcom_push(key='dataframe', value=df)
            print(f"[INFO] Dataset caricato correttamente: {csv_input} con {df.shape[0]} righe")
        except Exception as e:
            print(f"[ERROR] Errore nel caricamento del dataset: {e}")
            raise


    def generate_embeddings(**context):
        df = context['ti'].xcom_pull(key='dataframe')
        documents = df['cleaned_text'].astype(str).tolist()

        # Carica modello
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Genera embeddings
        embeddings = embedding_model.encode(documents, show_progress_bar=True)

        # Push in XCom: converte embeddings in lista per JSON
        context['ti'].xcom_push(key='embeddings', value=embeddings.tolist())
        context['ti'].xcom_push(key='documents', value=documents)

    def test_hdbscan_params(**context):
        documents = context['ti'].xcom_pull(key='documents')
        embeddings = context['ti'].xcom_pull(key='embeddings')

        param_list = [
            {'min_cluster_size': 10, 'min_samples': 5},
            {'min_cluster_size': 15, 'min_samples': 10},
            {'min_cluster_size': 20, 'min_samples': 15},
            {'min_cluster_size': 30, 'min_samples': 20},
            {'min_cluster_size': 40, 'min_samples': 25},
        ]

        best_score = -1
        best_params = None

        for params in param_list:
            clusterer = HDBSCAN(
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                metric='euclidean',
                cluster_selection_method='eom'
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            score = silhouette_score(embeddings, cluster_labels) if n_clusters > 1 else -1
            if score > best_score:
                best_score = score
                best_params = params

        context['ti'].xcom_push(key='best_params', value=best_params)


    def run_bertopic(**context):
        df = context['ti'].xcom_pull(key='dataframe')
        documents = context['ti'].xcom_pull(key='documents')
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # caricalo qui, perché non lo passiamo tramite XCom
        best_params = context['ti'].xcom_pull(key='best_params')

        hdbscan_model = HDBSCAN(
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom'
        )

        topic_model = BERTopic(
            language="english",
            min_topic_size=10,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            verbose=True,
        )

        topics, probs = topic_model.fit_transform(documents)
        df['Topic'] = topics
        df['Probability'] = probs

        topic_label_map = {
            topic: topic_model.get_topic(topic)[0][0] if topic != -1 else "outlier"
            for topic in topic_model.get_topics().keys()
        }
        df['Topic_Label'] = df['Topic'].map(lambda t: topic_label_map.get(t, "outlier"))

        # Save results and model
        df.to_csv("/opt/airflow/data/processed/bertopic_output.csv", index=False)

        # Salvare il modello con dill invece di topic_model.save()
        with open("/opt/airflow/models/bertopic_model.pkl", "wb") as f:
            dill.dump(topic_model, f)

    # Tasks

    load_dataset_preprocessed = PythonOperator(
        task_id='load_data_preprocessed',
        python_callable=load_data_preprocessed,
        provide_context=True
    )

    create_embeddings = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        provide_context=True
    )

    optimize_hdbscan = PythonOperator(
        task_id='test_hdbscan_params',
        python_callable=test_hdbscan_params,
        provide_context=True
    )

    bertopic_training = PythonOperator(
        task_id='run_bertopic',
        python_callable=run_bertopic,
        provide_context=True
    )

    # Define dependencies
    load_dataset_preprocessed >> create_embeddings >> optimize_hdbscan >> bertopic_training