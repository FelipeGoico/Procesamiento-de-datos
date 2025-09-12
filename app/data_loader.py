import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    return pd.read_csv("../data_processed.csv")


df = load_data()

print(df.head())


# ===========================
# Collage de histogramas
# ===========================
@st.cache_data
def get_graph(df):
    col_numericas = df.select_dtypes(
        include=[np.number]).columns.drop('label', errors='ignore')

    # Configurar grid de subplots
    n_cols = 3
    n_rows = int(np.ceil(len(col_numericas)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten()

    for i, col in enumerate(col_numericas):
        sns.histplot(df[col], bins=30, kde=True, color='#1f4e79', ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, col_numericas
