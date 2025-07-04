# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Charger les données
df = pd.read_csv("data.csv")

st.set_page_config(page_title="Dashboard Interactif", layout="wide")
st.title("📊 Dashboard Interactif de Données Temporelles")

st.sidebar.header("🔍 Filtres")
colonne = st.sidebar.selectbox("Sélectionnez une colonne à visualiser:", df.columns[1:])

fig = px.line(df, x='date', y=colonne, title=f"Évolution de {colonne}", markers=True)
st.plotly_chart(fig, use_container_width=True)

if st.checkbox("📁 Afficher les données brutes"):
    st.subheader("Données Brutes")
    st.dataframe(df)
