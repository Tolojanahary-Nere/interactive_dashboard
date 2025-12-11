import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class DataVisualizer:
    """
    Classe responsable de la création des graphiques.
    """
    
    @staticmethod
    def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str):
        """Crée un graphique de série temporelle interactif."""
        fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
        fig.update_layout(
            xaxis_title=x_col.capitalize(),
            yaxis_title=y_col.capitalize(),
            hovermode="x unified",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
        """Crée une heatmap de corrélation."""
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        fig.update_layout(title="Matrice de Corrélation")
        return fig

    @staticmethod
    def plot_prediction(df: pd.DataFrame, x_col: str, y_col: str, pred_col: str):
        """Affiche les données réelles vs les prédictions."""
        fig = go.Figure()
        
        # Données réelles
        fig.add_trace(go.Scatter(
            x=df[x_col], 
            y=df[y_col], 
            mode='markers', 
            name='Données Réelles',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Prédictions (Ligne de régression)
        fig.add_trace(go.Scatter(
            x=df[x_col], 
            y=df[pred_col], 
            mode='lines', 
            name='Prédiction (Régression Linéaire)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Modèle Prédictif : Réel vs Prédiction",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_white"
        )
        return fig
