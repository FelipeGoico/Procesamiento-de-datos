import streamlit as st
from app import st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import umap

# ============================================
# Banner estilo profesional
# ============================================
st.set_page_config(page_title="UMAP",
                   page_icon=":bar_chart:", layout="wide")

st.markdown("""
<div style="background-color:#1f4e79; padding: 18px; border-radius: 10px; text-align:center; color: white;">
    <h1 style="margin:0;">🗺️ Uniform Manifold Approximation and Projection (UMAP)</h1>
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
# UMAP 2D
# ============================================
st.markdown("### Proyección UMAP 2D del dataset")

umap_2d = umap.UMAP(n_components=2, random_state=42)
X_umap_2d = umap_2d.fit_transform(X_scaled)

df_umap_2d = pd.DataFrame(X_umap_2d, columns=['UMAP1', 'UMAP2'])
df_umap_2d['label'] = y.values

fig_2d = px.scatter(
    df_umap_2d, x='UMAP1', y='UMAP2', color='label',
    opacity=0.7, width=800, height=500,
    title="Proyección UMAP 2D"
)
st.plotly_chart(fig_2d, use_container_width=True)


# ============================================
# UMAP 3D
# ============================================
st.markdown("### Proyección UMAP 3D interactiva")

umap_3d = umap.UMAP(n_components=3, random_state=42)
X_umap_3d = umap_3d.fit_transform(X_scaled)

df_umap_3d = pd.DataFrame(X_umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
df_umap_3d['label'] = y.values

fig_3d = px.scatter_3d(
    df_umap_3d, x='UMAP1', y='UMAP2', z='UMAP3',
    color='label', opacity=0.7,
    width=800, height=600,
    title="Proyección UMAP 3D interactiva"
)
fig_3d.update_traces(marker=dict(size=4))
st.plotly_chart(fig_3d, use_container_width=True)


# ============================================
# Conclusión
# ============================================
st.markdown("""
### 🔎 Conclusiones

- UMAP permite visualizar la estructura de los datos preservando tanto relaciones locales como globales.
- Los clusters visibles reflejan agrupamientos naturales según el comportamiento de las vacas.
- Es más rápido que t-SNE y generalmente conserva mejor la estructura global del dataset.
- Esta visualización complementa PCA y t-SNE, ayudando a identificar patrones lineales y no lineales en los datos.
""")
