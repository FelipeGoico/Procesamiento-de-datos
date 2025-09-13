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
st.markdown("### Proyección UMAP 2D")

df_umap_2d = pd.DataFrame(X_umap_2d, columns=['UMAP1', 'UMAP2'])
df_umap_2d['label'] = y.values

fig_2d = px.scatter(
    df_umap_2d, x='UMAP1', y='UMAP2', color='label',
    opacity=0.7, width=800, height=650,
    title="Proyección UMAP 2D"
)
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("""
            
            <div style="text-align: justify; margin-top:20px;">
        
            #### Proyección UMAP 2D del Dataset
<ul>
<li><b>
Tipo:</b> Visualización no lineal de reducción dimensional (UMAP) en datos preprocesados IMU (~8000 muestras). Ejes: UMAP1 (X, estructura global/local), UMAP2 (Y, relaciones vecinales). Conserva topología global mejor que t-SNE, interactiva en Plotly.</li>

<li><b>Patrones por Clase:</b>
<ul>
<li><b>Rojo (Walking):</b> Clusters densos y extendidos en izquierda/centro (alta variabilidad en desplazamientos/aceleraciones), con formas irregulares.</li>
<li><b>Rosa (Resting):</b> Compacto en centro-derecha (baja dispersión, inactividad estática).</li>
<li><b>Azul (Grazing):</b> Pequeños grupos dispersos en derecha (ritmos moderados/estacionarios).</li>
<li><b>Azul claro (Miscellaneous):</b> Solapado y esparcido (transiciones/ruido heterogéneo).</li>
</li>
</ul>

<li><b>Separabilidad:</b> UMAP resalta clusters naturales y transiciones suaves.</li>
<li><b>Implicaciones:</b> Valida patrones en comportamientos vacas (e.g., walking vs. resting); outliers en UMAP1 negativo indican anomalías (estrés). Ideal para EDA escalable; en Streamlit, rota/zoom para detalles.</li>
            
""", unsafe_allow_html=True)

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

st.markdown("""
            #### Descripción del Gráfico
El gráfico representa una <b>proyección 3D interactiva de UMAP</b> (Uniform Manifold Approximation and Projection) del dataset Dataset. Muestra puntos dispersos en un espacio tridimensional (ejes: UMAP1, UMAP2, UMAP3), coloreados por categorías de comportamiento de vacas:
<ul>

<li><b>Miscellaneous behaviors:</b> Azul claro (puntos dispersos, posiblemente outliers).</li>
<li><b>Grazing:</b> Rosa (cluster central compacto).</li>
<li><b>Resting:</b> Rojo (agrupamiento denso en la parte superior).</li>
<li><b>Walking:</b> Azul (distribuido en áreas periféricas).</li>
</ul>
Se observan <b>clusters bien separados</b> que reflejan patrones naturales de actividad, con mayor densidad en las regiones de resting y grazing. El rango de ejes va de -5 a 15 aproximadamente, destacando la estructura no lineal preservada por UMAP para visualización de datos de sensores IMU.
""", unsafe_allow_html=True)
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
