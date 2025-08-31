# FinSent — Sentiment Analysis per Insight Finanziari Automatizzati

> **Descrizione breve:** Pipeline MLOps per la sentiment analysis di notizie finanziarie in lingua inglese relative a società quotate nell'indice OMX Helsinki.  
> **Tagline:** *Da testo finanziario a insight in produzione.*

---

## 📖 Abstract
FinSent affronta il compito della sentiment analysis applicata a notizie finanziarie. Il sistema automatizza preprocessing, generazione di embedding, addestramento, valutazione comparativa e deployment del modello. Ogni componente è containerizzato con **Docker** e orchestrato tramite **Apache Airflow**. Il modello finale è esposto tramite una REST API sviluppata con **FastAPI**, che consente inferenze in tempo reale. Versioning e tracciabilità sono garantiti da **DVC** e DAG dedicati.  
L’impiego di modelli transformer pre-addestrati ha consentito di raggiungere un’accuratezza vicina al **99%** dopo il fine-tuning.

---

## 1. Il Problema
L’analisi automatica del sentiment nelle notizie finanziarie è cruciale: articoli e report influenzano i mercati.  
Obiettivo del progetto è costruire un sistema in grado di classificare automaticamente il sentiment (positivo, negativo, neutrale), supportando decisioni di investimento più rapide e informate.

---

## 2. Metodologia (NLP)

### Dataset
- **FinancialPhraseBank** (~5000 frasi annotate manualmente da 16 valutatori).  
- Classi sbilanciate: **Neutral 2879**, **Positive 1363**, **Negative 604**.

### 2.1 Pre-processing e Analisi dei Dati
Due approcci:
- **Light**: normalizzazione minima.  
- **Aggressive**: rimozione più forte di stopword e ridondanze.  

Entrambi hanno ridotto la lunghezza media delle frasi e migliorato la qualità degli input.

### 2.2 Modelli e Risultati
- **Embedding testati:** Word2Vec, BERT, SBERT, FinBERT.  
- **Classificatori supervisionati:** SVM, Random Forest, MLP, Logistic Regression.
- **Modelli pre-addestrati:**
  [deberta-v3-ft-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/deberta-v3-ft-financial-news-sentiment-analysis), [distilroberta-finetuned-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis), [financial-roberta-large-sentiment](https://huggingface.co/soleimanian/financial-roberta-large-sentiment), [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone).



**Baseline:** SVM con embedding FinBERT → **84.9%** accuracy.  
**Avanzato:** Fine-tuning di **DeBERTa-v3-ft** → accuracy vicina al **99%**.

### 2.3 Topic Modeling
- **BERTopic** integrato per estrarre ~113 topic.  
- Risultati: contenuti neutrali distribuiti su molti argomenti; quelli negativi concentrati su pochi, spesso collegati a eventi specifici.

---

## 3. Architettura MLOps
- **Orchestrazione:** Apache Airflow con DAG modulari.  
- **Containerizzazione:** Docker e Docker Compose.  
- **API di Inferenza:** FastAPI per predizioni real-time.  
- **Versioning:** DVC per dataset e modelli.  
- **Tracciabilità:** esperimenti e metriche gestiti tramite DAG dedicati.

---

## ✨ Caratteristiche
- [x] Pipeline MLOps end-to-end (preprocessing, embedding, training, valutazione, topic modeling, deployment)  
- [x] Containerizzazione con Docker  
- [x] Orchestrazione workflow con Airflow  
- [x] API REST con FastAPI  
- [x] Versioning dati e modelli con DVC  
- [ ] Estensione multilingua e multi-mercato  

> **Perché usarlo?** Automazione completa, accuratezza elevata, API pronte per l’integrazione.

---

## 🚀 Running the Project with Docker and Airflow
**Stack principale:**
- Python 3.10+, FastAPI, Uvicorn  
- Apache Airflow 2.x  
- Hugging Face Transformers, scikit-learn, sentence-transformers, BERTopic, UMAP, NLTK  
- DVC  
- Docker & Docker Compose  

**Requisiti minimi:**
- Docker + Docker Compose  
- Python 3.10+  
- 8 GB RAM (GPU NVIDIA opzionale per fine-tuning transformer)

---

### 🔧 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/IreneGaita/FinSent.git
   cd repo
   ```
2. **Make sure Docker is running**
   
   Start Docker Desktop and wait until it's fully operational.
3. **Navigate to the `airflow/`folder (if applicable)**
    ```bash
    cd airflow
    ```
4. **Build and start the containers**
   
   Run the following command:
   ```bash
    docker-compose up --build
    ```
     > Use `docker-compose up -d` to start the containers in the background.
     
### 🌐 **Access the Airflow Web Interface**
5. **Open your browser and go to:**
   ```bash
    http://localhost:8080
    ```
   > ⚠️ Check the correct port in the docker-compose.yml file under the ports: section (e.g., 8080:8080 or 8081:8080).
   
6. **Login credentials**
   Default credentials are usually:
   - **Username:** airflow
   - **Password:** airflow
     
  > ⚠️ You can confirm or override these values in the docker-compose.yml file under the environment section.
   
### 🛑 **Shutting Down the Project**
To stop the containers, run:
  ```bash
  docker-compose down
  ```
---

## 💻 Esempi di utilizzo API

1.  **Richiesta con `curl`**

    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"text": "The company reported record profits this quarter."}'
    ```

    **Risposta attesa:**

    ```json
    {
      "text": "The company reported record profits this quarter.",
      "sentiment": "Positive",
      "confidence": 0.97
    }
    ```

2.  **Richiesta con Python (`requests`)**

    ```python
    import requests

    url = "http://localhost:8000/predict"
    payload = {"text": "Stock prices fell sharply after the announcement."}

    response = requests.post(url, json=payload)
    print(response.json())
    ```

    **Output atteso:**

    ```json
    {
      "text": "Stock prices fell sharply after the announcement.",
      "sentiment": "Negative",
      "confidence": 0.95
    }
    ```
---

## 📚 Documentazione

La documentazione completa del progetto è disponibile nella cartella [`docs/`](./docs):

- [NLP.pdf](./docs/NLP.pdf) — Dettagli su preprocessing, modelli e analisi NLP.
- [MLOps.pdf](./docs/MLOps.pdf) — Architettura MLOps, orchestrazione e pipeline con Airflow.

---

## 👥 Author & Contact

* **Irene Gaita**
    * 📧 Email: igaita3107@gmail.com
    * 🔗 LinkedIn: [linkedin.com/in/irene-gaita-4ba32822b](https://www.linkedin.com/in/irene-gaita-4ba32822b)

* **Mario Cicalese**
    * 📧 Email: m.cicalese21@studenti.unisa.it
    * 🔗 LinkedIn: [linkedin.com/in/mario-cicalese-5b26a5283](https://www.linkedin.com/in/mario-cicalese-5b26a5283)

---

## 🙏 Riconoscimenti

* **Dataset:** [FinancialPhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
