from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import dill
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

# Default args
default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('bertopic_topic_modeling_dynamic',
         schedule_interval=None,
         default_args=default_args,
         catchup=False) as dag:
    # --- Task Functions ---

    def load_data_preprocessed(csv_input, pipeline_type='full', **context):
        import os
        try:
            assert os.path.exists(csv_input), f"File non trovato: {csv_input}"
            df = pd.read_csv(csv_input)
            if 'cleaned_text' not in df.columns:
                raise ValueError("Colonna 'cleaned_text' mancante nel CSV")
            context['ti'].xcom_push(key=f'dataframe_{pipeline_type}', value=df)
            print(f"[INFO] Dataset caricato correttamente: {csv_input} con {df.shape[0]} righe")
        except Exception as e:
            print(f"[ERROR] Errore nel caricamento del dataset: {e}")
            raise


    def generate_embeddings(pipeline_type='full', **context):
        key = f'dataframe_{pipeline_type}'
        df = context['ti'].xcom_pull(key=key)

        if df is None:
            raise ValueError(f"Nessun dataframe trovato per la pipeline '{pipeline_type}'")

        documents = df['cleaned_text'].astype(str).tolist()
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(documents, show_progress_bar=True)

        context['ti'].xcom_push(key=f'embeddings_{pipeline_type}', value=embeddings.tolist())
        context['ti'].xcom_push(key=f'documents_{pipeline_type}', value=documents)


    def test_hdbscan_params(pipeline_type='full', **context):
        documents = context['ti'].xcom_pull(key=f'documents_{pipeline_type}')
        embeddings = context['ti'].xcom_pull(key=f'embeddings_{pipeline_type}')

        if embeddings is None or documents is None:
            raise ValueError(f"Missing embeddings or documents for pipeline: {pipeline_type}")

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

        context['ti'].xcom_push(key=f'best_params_{pipeline_type}', value=best_params)


    def run_bertopic(pipeline_type='full', **context):
        df = context['ti'].xcom_pull(key=f'dataframe_{pipeline_type}')
        documents = context['ti'].xcom_pull(key=f'documents_{pipeline_type}')
        best_params = context['ti'].xcom_pull(key=f'best_params_{pipeline_type}')

        if df is None or documents is None or best_params is None:
            raise ValueError(f"Missing data for pipeline: {pipeline_type}")

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

        output_csv_path = f"/opt/airflow/data/processed/bertopic_output_{pipeline_type}.csv"
        model_path = f"/opt/airflow/models/bertopic_model_{pipeline_type}.pkl"

        df.to_csv(output_csv_path, index=False)
        with open(model_path, "wb") as f:
            dill.dump(topic_model, f)


    # --- Dynamic Tasks per pipeline_type ---
    for pipeline_type in ['full', 'light']:
        csv_map = {
            'full': '/opt/airflow/data/processed/preprocessed_data.csv',
            'light': '/opt/airflow/data/processed/light_preprocessed_data.csv'
        }
        csv_input = csv_map[pipeline_type]

        load_task = PythonOperator(
            task_id=f'load_data_{pipeline_type}',
            python_callable=load_data_preprocessed,
            op_kwargs={
                'csv_input': csv_map[pipeline_type],
                'pipeline_type': pipeline_type
            },
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

        load_task >> embeddings_task >> hdbscan_task >> bertopic_task