import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


DATA_PATH = "../sentiment_data/preprocessed_data.csv"
VECTORIZER_PATH = "../models/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = "../sentiment_data/X_tfidf.npz"

def load_data(path):
    df = pd.read_csv(path)
    return df['cleaned_text'], df['label']

#Creazione del vettorizzatore TF-IDF
def create_tfidf_features(text_series, max_features=5000, ngram_range=(1, 2), min_df=5):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'
    )
    X_tfidf = vectorizer.fit_transform(text_series)
    return X_tfidf, vectorizer

#Salvataggio del vettorizzatore
def save_vectorizer(vectorizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)

#Salvataggio della matrice TF-IDF
def save_tfidf_matrix(X_tfidf, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sparse.save_npz(path, X_tfidf)

    # Salva le etichette in formato .csv
    y.to_csv("../sentiment_data/y.csv", index=False)

# === Main ===
if __name__ == "__main__":
    print("Caricamento dati...")
    X_text, y = load_data(DATA_PATH)

    print("Creazione della matrice TF-IDF...")
    X_tfidf, vectorizer = create_tfidf_features(X_text)

    print("Salvataggio del vettorizzatore e della matrice...")
    save_vectorizer(vectorizer, VECTORIZER_PATH)
    save_tfidf_matrix(X_tfidf, TFIDF_MATRIX_PATH)

    print(f"\n TF-IDF completato.")
    print(f"   - Matrice shape: {X_tfidf.shape}")
    print(f"   - Vettorizzatore salvato in: {VECTORIZER_PATH}")
    print(f"   - Matrice TF-IDF salvata in: {TFIDF_MATRIX_PATH}")

    print(X_tfidf[:5].todense())