import streamlit as st
import pandas as pd
from src.data_loader import DataLoader
from src.analysis import DataAnalyzer
from src.visualization import DataVisualizer

# Configuration de la page
st.set_page_config(
    page_title="Analytics Dashboard Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look plus pro
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("🚀 Analytics Pro")
    st.sidebar.markdown("---")
    
    # Chargement des données
    data_loader = DataLoader("data.csv")
    df = data_loader.load_data()
    
    if df.empty:
        st.error("Impossible de charger les données. Vérifiez le fichier data.csv.")
        return

    # Navigation
    page = st.sidebar.radio("Navigation", ["Vue d'ensemble", "Analyse Exploratoire (EDA)", "Machine Learning"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Cette application permet d'analyser des séries temporelles, d'explorer les corrélations entre variables "
        "et de réaliser des prédictions simples grâce au Machine Learning."
    )

    # Instanciation des classes d'analyse et de visualisation
    analyzer = DataAnalyzer(df)
    visualizer = DataVisualizer()

    # Fonction utilitaire pour l'affichage sécurisé
    def safe_dataframe(data):
        """Affiche un dataframe en convertissant les dates en str pour éviter les erreurs PyArrow."""
        if isinstance(data, pd.DataFrame):
            df_show = data.copy()
            # Convertir les colonnes datetime et object contenant des dates en string
            for col in df_show.columns:
                if pd.api.types.is_datetime64_any_dtype(df_show[col]) or 'date' in col.lower():
                    df_show[col] = df_show[col].astype(str)
            st.dataframe(df_show, width='stretch')
        else:
            st.dataframe(data, width='stretch')

    if page == "Vue d'ensemble":
        st.title("📊 Vue d'ensemble des Données")
        st.markdown("Bienvenue sur ce tableau de bord interactif d'analyse de données temporelles.")
        
        # Métriques clés (KPIs)
        col1, col2, col3 = st.columns(3)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0] # Par défaut la première colonne numérique
            with col1:
                st.metric("Total Enregistrements", len(df))
            with col2:
                st.metric(f"Moyenne {target_col}", f"{df[target_col].mean():.2f}")
            with col3:
                st.metric(f"Max {target_col}", f"{df[target_col].max():.2f}")

        st.markdown("### 📈 Évolution Temporelle")
        col_to_plot = st.selectbox("Choisir la métrique à visualiser :", numeric_cols)
        
        if 'date' in df.columns:
            fig = visualizer.plot_time_series(df, 'date', col_to_plot, f"Évolution de {col_to_plot} dans le temps")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Aucune colonne 'date' trouvée pour la série temporelle.")

        with st.expander("Voir les données brutes"):
            safe_dataframe(df)

    elif page == "Analyse Exploratoire (EDA)":
        st.title("🔍 Analyse Exploratoire des Données")
        
        tab1, tab2 = st.tabs(["Statistiques Descriptives", "Corrélations"])
        
        with tab1:
            st.subheader("Statistiques Globales")
            safe_dataframe(analyzer.get_basic_stats())
            
        with tab2:
            st.subheader("Matrice de Corrélation")
            corr_matrix = analyzer.get_correlations()
            fig_corr = visualizer.plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, width='stretch')
            
            st.markdown("**Interprétation :** Une valeur proche de 1 indique une forte corrélation positive, tandis qu'une valeur proche de -1 indique une forte corrélation négative.")

    elif page == "Machine Learning":
        st.title("🤖 Prédiction (Machine Learning)")
        st.markdown("Démonstration d'un pipeline simple de Machine Learning : **Régression Linéaire**.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration du Modèle")
            numeric_cols = df.select_dtypes(include=['number']).columns
            target = st.selectbox("Variable Cible (y)", numeric_cols, index=0)
            feature = st.selectbox("Variable Explicative (X)", ['date'] + list(numeric_cols), index=0)
            
            train_btn = st.button("Entraîner le Modèle")
            
        with col2:
            if train_btn:
                with st.spinner("Entraînement du modèle en cours..."):
                    try:
                        model, mse, r2, df_pred = analyzer.train_predict_model(target, feature)
                        
                        st.success("Modèle entraîné avec succès !")
                        
                        # Affichage des métriques
                        m1, m2 = st.columns(2)
                        m1.metric("MSE (Erreur Quadratique Moyenne)", f"{mse:.2f}")
                        m2.metric("R² Score (Précision)", f"{r2:.4f}")
                        
                        # Visualisation
                        st.subheader("Résultats de la Prédiction")
                        fig_pred = visualizer.plot_prediction(df_pred, feature, target, 'prediction')
                        st.plotly_chart(fig_pred, width='stretch')
                        
                        st.info("Note : Ce modèle est une régression linéaire simple à des fins de démonstration. Pour des séries temporelles complexes, des modèles comme Prophet ou LSTM seraient plus appropriés.")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement : {e}")

if __name__ == "__main__":
    main()
