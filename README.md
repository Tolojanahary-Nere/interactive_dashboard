# 📊 Analytics Dashboard Pro

Ce projet est une démonstration de compétences en **Backend Python** et **Data Science**, conçu pour illustrer les meilleures pratiques de développement et d'analyse de données.

## 🎯 Objectifs du Projet

*   **Architecture Modulaire** : Structure de code propre et maintenable (`src/` avec séparation des responsabilités).
*   **Data Science** : Pipeline complet incluant EDA (Analyse Exploratoire) et Machine Learning (Régression Linéaire).
*   **Visualisation Avancée** : Utilisation de Plotly pour des graphiques interactifs et professionnels.
*   **Performance** : Utilisation du cache Streamlit pour optimiser le chargement des données.

## 🛠️ Stack Technique

*   **Langage** : Python 3.12+
*   **Framework Web** : Streamlit
*   **Manipulation de Données** : Pandas, NumPy
*   **Machine Learning** : Scikit-Learn
*   **Visualisation** : Plotly Express & Graph Objects

## 🚀 Installation et Lancement

1.  **Cloner le dépôt** :
    ```bash
    git clone https://github.com/Tolojanahary-Nere/interactive_dashboard.git
    cd interactive_dashboard
    ```

2.  **Créer un environnement virtuel** (recommandé) :
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Linux/Mac
    # venv\Scripts\activate  # Sur Windows
    ```

3.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancer l'application** :
    ```bash
    ./run.sh
    # ou
    streamlit run app.py
    ```

## 📂 Structure du Projet

```
interactive_dashboard/
├── app.py              # Point d'entrée de l'application
├── run.sh              # Script de lancement rapide
├── requirements.txt    # Dépendances du projet
├── data/               # Dossier de données
│   └── data.csv
└── src/                # Code source modulaire
    ├── __init__.py
    ├── data_loader.py  # Gestion du chargement des données (avec cache)
    ├── analysis.py     # Logique métier et Machine Learning
    └── visualization.py # Création des graphiques interactifs
```

## 👤 Auteur

**Tolojanahary Nere** - *Data Scientist & Backend Python Developer*
