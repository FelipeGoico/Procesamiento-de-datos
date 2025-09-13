import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import get_full_data, preprocess_data, compute_umap, init_data, load_umap_data, set_global_config

set_global_config()

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

# Inicializar datos y gráficos
init_data()
load_umap_data()

# ============================================
# Datos preprocesados y cacheados
# ============================================

if "X_umap_2d" not in st.session_state or "X_umap_3d" not in st.session_state or "umap_y" not in st.session_state:
    if "df" not in st.session_state:
        init_data()        # carga df, muestra y gráficos
    load_umap_data()       # calcula UMAP solo si no existe

# Recuperar de session_state
X_umap_2d = st.session_state.X_umap_2d
X_umap_3d = st.session_state.X_umap_3d
y = st.session_state.umap_y


# ============================================
# Graficar UMAP 2D
# ============================================
st.markdown("### Proyección UMAP 2D del dataset")

df_umap_2d = pd.DataFrame(X_umap_2d, columns=['UMAP1', 'UMAP2'])
df_umap_2d['label'] = y.values

fig_2d = px.scatter(
    df_umap_2d, x='UMAP1', y='UMAP2', color='label',
    opacity=0.7, width=800, height=650,
    title="Proyección UMAP 2D"
)
st.plotly_chart(fig_2d, use_container_width=True)

# ============================================
# Graficar UMAP 3D
# ============================================
st.markdown("### Proyección UMAP 3D interactiva")

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
st.divider()
if st.button("Volver a la Página Principal"):
    st.switch_page("app.py")
