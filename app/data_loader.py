import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ===========================
# Carga de datos
# ===========================
@st.cache_data
def load_data(path="../data_processed.csv"):
    return pd.read_csv(path)

df = load_data()
st.write("Datos cargados:", df.shape)
st.dataframe(df.head())

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

# ===========================
# Preprocesamiento: train/test + KNN + Escalado
# ===========================
@st.cache_data
def prepare_data(df, label_col='label', test_size=0.2, random_state=42):
    feature_cols = [col for col in df.columns if col != label_col and df[col].dtype in [np.float64, np.int64]]
    X = df[feature_cols]
    y = df[label_col]

    # Dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Imputación KNN solo sobre train
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    # Escalado solo sobre train
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_cols)

    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

# ===========================
# Función para obtener datos preprocesados
# ===========================
def get_preprocessed_data():
    return prepare_data(df)
