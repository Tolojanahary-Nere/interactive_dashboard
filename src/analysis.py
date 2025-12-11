import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class DataAnalyzer:
    """
    Classe responsable de l'analyse des données et du Machine Learning.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_basic_stats(self) -> pd.DataFrame:
        """Retourne les statistiques descriptives de base."""
        return self.df.describe()

    def get_correlations(self) -> pd.DataFrame:
        """Calcule la matrice de corrélation pour les colonnes numériques."""
        return self.df.select_dtypes(include=[np.number]).corr()

    def train_predict_model(self, target_col: str, feature_col: str = 'date_ordinal') -> Tuple[Any, float, float, pd.DataFrame]:
        """
        Entraîne un modèle de régression linéaire simple.
        Si la feature est une date, elle est convertie en ordinal.
        
        Returns:
            Tuple contenant le modèle, le MSE, le R2 score et les prédictions.
        """
        df_ml = self.df.copy()
        
        # Préparation des features
        if 'date' in str(df_ml[feature_col].dtype): # Si c'est une date
             df_ml['date_ordinal'] = pd.to_datetime(df_ml[feature_col]).map(pd.Timestamp.toordinal)
             X = df_ml[['date_ordinal']]
        else:
             X = df_ml[[feature_col]]
             
        y = df_ml[target_col]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Prédictions sur l'ensemble pour la visualisation
        df_ml['prediction'] = model.predict(X)
        
        return model, mse, r2, df_ml
