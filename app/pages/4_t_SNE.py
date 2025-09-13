import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import preprocess_data, compute_tsne, load_umap_data, init_tsne, init_data, set_global_config

set_global_config()


# ============================================
# Banner estilo profesional
# ============================================
st.set_page_config(
    page_title="t-SNE",
    page_icon=":bar_chart:",
    layout="wide"
)
# Inicializar datos y gráficos
init_data()
st.markdown("""
<div style="background-color:#1f4e79; padding: 18px; border-radius: 10px; text-align:center; color: white;">
    <h1 style="margin:0;">📉 t-Distributed Stochastic Neighbor Embedding (t-SNE)</h1>
    <p style="margin:0; font-size:18px;">Reducción de dimensionalidad y visualización del dataset <i>db-cow-walking-IoT</i></p>
</div>
""", unsafe_allow_html=True)


# ============================================
# Cargar datos preprocesados o muestreados
# ============================================
# Asegura que los datos y t-SNE estén cargados
if "tsne_2d" not in st.session_state or "tsne_3d" not in st.session_state or "tsne_y" not in st.session_state:
    if "df" not in st.session_state:
        init_data()      # carga df, muestra y gráficos
    init_tsne()          # calcula t-SNE solo si no existe

X_tsne_2d = st.session_state.tsne_2d
X_tsne_3d = st.session_state.tsne_3d
y = st.session_state.tsne_y
# ============================================
# t-SNE 2D
# ============================================
st.markdown("### Proyección t-SNE 2D del dataset")

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
- Esta visualización complementa PCA y UMAP, ayudando a identificar patrones que podrían no ser lineales.
""")
st.divider()
if st.button("Volver a la Página Principal"):
    st.switch_page("app.py")
load_umap_data()
