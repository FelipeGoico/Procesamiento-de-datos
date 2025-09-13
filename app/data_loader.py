import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def set_global_config():
    st.set_page_config(
        page_title="Presentación del Proyecto de investigación",
        initial_sidebar_state="collapsed",
        page_icon=":bar_chart:",
        layout="wide"
    )

# ===========================
# Carga de datos crudos
# ===========================


@st.cache_data
def get_full_data(path="../data_processed.csv"):
    """Carga el dataset completo desde disco"""
    return pd.read_csv(path)


# ===========================
# Muestreo balanceado
# ===========================
@st.cache_data
def get_sample_data(df_total, n_muestra=8000):
    """Devuelve una muestra balanceada de tamaño n_muestra"""
    return df_total.groupby("label", group_keys=False).sample(
        frac=n_muestra / len(df_total), random_state=42
    )


# ===========================
# Collage de histogramas
# ===========================


@st.cache_data
def get_graph(df):
    """Genera collage de histogramas para todas las variables numéricas"""
    col_numericas = df.select_dtypes(
        include=[np.number]).columns.drop("label", errors="ignore")

    # Configurar grid
    n_cols = 3
    n_rows = int(np.ceil(len(col_numericas) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(col_numericas):
        sns.histplot(df[col], bins=30, kde=True, color="#1f4e79", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Eliminar ejes vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, col_numericas


# ===========================
# Preprocesamiento train/test
# ===========================
@st.cache_data
def prepare_data(df, label_col="label", test_size=0.2, random_state=42):
    """Prepara datos con imputación KNN + escalado"""
    feature_cols = [
        col for col in df.columns
        if col != label_col and df[col].dtype in [np.float64, np.int64]
    ]

    X = df[feature_cols]
    y = df[label_col]

    # Dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Imputación KNN
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=feature_cols)
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=feature_cols)

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(
        X_train_imputed), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(
        X_test_imputed), columns=feature_cols)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ===========================
# Wrapper de preprocesamiento
# ===========================
def get_preprocessed_data():
    """Devuelve datos ya preprocesados desde session_state"""
    return st.session_state.preprocessed


@st.cache_data
def preprocess_data(df, max_samples=None):
    """Imputa, escala y opcionalmente muestrea"""
    feature_cols = [col for col in df.columns if col !=
                    'label' and df[col].dtype in [np.float64, np.int64]]
    X = df[feature_cols]
    y = df['label']

    # Imputación
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    # Escalado
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(
        X_imputed), columns=feature_cols)

    # Muestreo si es necesario
    if max_samples and len(X_scaled) > max_samples:
        sample_idx = np.random.choice(
            len(X_scaled), size=max_samples, replace=False)
        X_scaled = X_scaled.iloc[sample_idx]
        y = y.iloc[sample_idx]

    return X_scaled, y, feature_cols

# =========================
# t-SNE
# =========================


@st.cache_data
def compute_tsne(X, n_components=2, perplexity=30, learning_rate=200, random_state=42):
    """Calcula TSNE y devuelve la proyección"""
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="random",
        random_state=random_state
    )
    return tsne_model.fit_transform(X)

# =========================
# UMAP
# =========================


@st.cache_data
def compute_umap(X, n_components=2, random_state=42):
    """Calcula UMAP"""
    from umap import UMAP
    reducer = UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)

# ===========================
# Inicialización global
# ===========================


def init_data():
    if "df" not in st.session_state:
        df_total = get_full_data()
        st.session_state.df = get_sample_data(df_total)

    if "graphs" not in st.session_state:
        st.session_state.graphs = get_graph(st.session_state.df)

    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = prepare_data(st.session_state.df)


def init_tsne():
    if "tsne_2d" not in st.session_state or "tsne_3d" not in st.session_state:
        X_scaled, y, _ = preprocess_data(st.session_state.df, max_samples=8000)
        st.session_state.tsne_2d = compute_tsne(X_scaled, n_components=2)
        st.session_state.tsne_3d = compute_tsne(X_scaled, n_components=3)
        st.session_state.tsne_y = y


def load_umap_data():
    if "umap_y" not in st.session_state or "X_umap_2d" not in st.session_state or "X_umap_3d" not in st.session_state:
        df = st.session_state.df
        X_scaled, y, feature_cols = preprocess_data(df, max_samples=8000)
        st.session_state.X_umap_2d = compute_umap(X_scaled, n_components=2)
        st.session_state.X_umap_3d = compute_umap(X_scaled, n_components=3)
        st.session_state.umap_y = y
