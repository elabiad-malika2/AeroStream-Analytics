#  AeroStream : Système Intelligent de Classification d'Avis Clients

##  Présentation du Projet
AeroStream est une solution complète de traitement de données (Pipeline End-to-End) permettant d'analyser en temps réel le sentiment des clients des compagnies aériennes américaines. Le projet utilise le Machine Learning pour classifier les avis et un tableau de bord pour visualiser les performances des compagnies.

---

##  Architecture Technique

Le projet s'articule autour de 4 piliers majeurs :

1.  **Pipeline Batch (Entraînement) :** 
    *   Source : Dataset `7Xan7der7/usairlinesentiment` (Hugging Face).
    *   Stockage Vectoriel : **ChromaDB** pour la gestion des embeddings.
    *   Modèle : NLP avec `Sentence Transformers` (cardiffnlp/twitter-roberta-base-sentiment).
2.  **Service API (FastAPI) :**
    *   **Endpoint 1 :** Récupération des données brutes.
    *   **Endpoint 2 :** Intelligence (Nettoyage + Embedding + Prédiction du sentiment).
3.  **Orchestration (Airflow) :**
    *   Automatisation du flux de données toutes les minutes.
    *   Récupération des prédictions via l'API et stockage dans **PostgreSQL**.
4.  **Visualisation (Streamlit) :**
    *   Tableau de bord interactif connecté à PostgreSQL (pgAdmin).
    *   Calcul des KPIs et agrégations en temps réel.

---

##  Stack Technologique
*   **NLP & ML :** Python, Scikit-learn, Sentence-Transformers.
*   **Vector DB :** ChromaDB.
*   **API Framework :** FastAPI & Uvicorn.
*   **Orchestrateur :** Apache Airflow.
*   **Base de Données :** PostgreSQL (pgAdmin).
*   **Dashboard :** Streamlit & Plotly.

---

##  Structure du Projet et Utilisation

### 1. Partie Batch & Modèle
Avant de lancer le streaming, le script de batch prépare les données :
- Téléchargement du dataset.
- Nettoyage (Regex pour supprimer les @mentions et URLs).
- Stockage des vecteurs dans **ChromaDB**.
- Entraînement et sauvegarde du meilleur classifieur.

