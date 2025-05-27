FROM apache/airflow:2.9.1

USER root

# Installa le dipendenze di sistema richieste da bertopic
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia il requirements.txt nel container
COPY ./dags/requirements.txt /requirements.txt

# Torna all'utente airflow per installare i pacchetti Python
USER airflow

# Installa le dipendenze Python come utente airflow
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /requirements.txt
