from app import st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# ============================================
# Banner estilo profesional
# ============================================
st.set_page_config(page_title="t-SNE",
                   page_icon=":bar_chart:", layout="wide")

st.markdown("""
<div style="background-color:#1f4e79; padding: 18px; border-radius: 10px; text-align:center; color: white;">
    <h1 style="margin:0;">📉 t-Distributed Stochastic Neighbor Embedding (t-SNE)</h1>
    <p style="margin:0; font-size:18px;">Reducción de dimensionalidad y visualización del dataset <i>db-cow-walking-IoT</i></p>
</div>
""", unsafe_allow_html=True)


# ============================================
# Datos
# ============================================
df = st.session_state.df

feature_cols = [col for col in df.columns if col != 'label' and df[col].dtype in [np.float64, np.int64]]
X = df[feature_cols]
y = df['label']

# Imputación y escalado
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols)


# ============================================
# t-SNE 2D
# ============================================
st.markdown("### Proyección t-SNE 2D del dataset")

tsne_2d = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

df_tsne_2d = pd.DataFrame(X_tsne_2d, columns=['tSNE1', 'tSNE2'])
df_tsne_2d['label'] = y.values

fig_2d = px.scatter(
    df_tsne_2d, x='tSNE1', y='tSNE2', color='label',
    opacity=0.7, width=800, height=500,
    title="Proyección t-SNE 2D (perplexity=30, learning_rate=200)"
)
st.plotly_chart(fig_2d, use_container_width=True)


# ============================================
# t-SNE 3D
# ============================================
st.markdown("### Proyección t-SNE 3D interactiva")

tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

df_tsne_3d = pd.DataFrame(X_tsne_3d, columns=['tSNE1', 'tSNE2', 'tSNE3'])
df_tsne_3d['label'] = y.values

fig_3d = px.scatter_3d(
    df_tsne_3d, x='tSNE1', y='tSNE2', z='tSNE3',
    color='label', opacity=0.7,
    width=800, height=600,
    title="Proyección t-SNE 3D interactiva"
)
fig_3d.update_traces(marker=dict(size=4))
st.plotly_chart(fig_3d, use_container_width=True)


# ============================================
# Conclusión
# ============================================
st.markdown("""
### 🔎 Conclusiones

- t-SNE permite visualizar la estructura de los datos de sensores en **2D y 3D**, preservando relaciones locales.
- Los clusters visibles reflejan agrupamientos naturales según el comportamiento de las vacas.
- Ajustando los hiperparámetros `perplexity` y `learning_rate`, se puede enfatizar la preservación global o local de la estructura.
- Esta visualización complementa PCA, ayudando a identificar patrones que podrían no ser lineales.
""")
