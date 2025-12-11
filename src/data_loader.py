import pandas as pd
import streamlit as st
from typing import Optional

class DataLoader:
    """
    Classe responsable du chargement et du pré-traitement des données.
    Utilise le cache de Streamlit pour optimiser les performances.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    @st.cache_data
    def load_data(_self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier CSV.
        
        Returns:
            pd.DataFrame: Le dataframe chargé et nettoyé.
        """
        try:
            df = pd.read_csv(_self.file_path)
            # Conversion automatique des colonnes de date si détectées
            for col in df.columns:
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return pd.DataFrame()
