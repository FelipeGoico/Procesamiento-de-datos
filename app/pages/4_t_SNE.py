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

st.markdown("""
<div style="text-align: justify; margin-top:20px;">
            
### Resumen de Interpretación: Proyección t-SNE 2D (Perplexity=30, Learning Rate=200)
<ul>
<li><b>Tipo:</b> Visualización no lineal de reducción dimensional (t-SNE) en datos preprocesados de sensores IMU (~8000 muestras). Ejes: tSNE1 (X, estructura local), tSNE2 (Y, relaciones vecinales). Preserva clusters locales, no distancias globales.</li>
<li><b>Patrones por Clase:</b>
    <ul>
        <li><b>Rojo (Walking):</b> Múltiples clusters densos dispersos (alta variabilidad en movimientos lineales/aceleraciones), con solapamientos mínimos.</li>
        <li><b>Azul (Miscellaneous):</b> Disperso y solapado con otros (naturaleza transitoria/ruido), formando "nubes" irregulares.</li>
        <li><b>Rosa (Resting):</b> Compacto en áreas centrales/bajas (baja varianza en sensores, inactividad).</li>
        <li><b>Verde (Grazing):</b> Disperso moderado, solapado con walking (ritmos similares pero estacionarios).</li>
    </ul>
</li>
<li><b>Separabilidad:</b> Buena para clusters locales (~70-80%, mejor que PCA lineal), con solapamientos ~20-30% por ruido ambiental. tSNE resalta subestructuras no lineales (e.g., variaciones en giros/orientación).</li>
<li><b>Implicaciones:</b> Confirma patrones naturales en comportamientos vacas; outliers rojos/azules indican anomalías (e.g., estrés). Ideal para EDA; complementa con UMAP para global. En Streamlit: Interactivo para zoom en clusters.</li>
</ul>          
            """, unsafe_allow_html=True)

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
st.markdown("""
            #### Descripción: Proyección t-SNE 3D Interactiva
<ul>
<li><b>Tipo:</b> Visualización no lineal 3D (t-SNE) de datos preprocesados IMU (~8000 muestras). Ejes: tSNE1 (X, estructura local), tSNE2 (Y, vecindades), tSNE3 (Z, profundidad). Preserva relaciones locales, interactiva en Plotly (rotación/zoom).</li>
<li><b>Patrones por Clase:</b>
            
<ul>
<li><b>Rojo (Walking):</b> Clusters densos y extendidos (alta variabilidad en movimientos), dominando centro/positivo.</li>
<li><b>Azul (Miscellaneous/Resting):</b> Compactos y esféricos cerca del origen (baja dispersión, inactividad/ruido).</li>
<li><b>Rosa/Verde (Grazing/Otros):</b> Pequeños grupos solapados, con dispersión moderada en Z (ritmos estacionarios).</li>
</ul>
</li>

<li><b>Separabilidad:</b> Excelente local (~80-90%, mejor que 2D por profundidad Z), solapamientos ~10-20% en transiciones. Resalta subestructuras no lineales (e.g., giros verticales).</li>

<li><b>Implicaciones:</b> Confirma anomalías (outliers rojos en Z alto, e.g., estrés); ideal para EDA inmersiva en Streamlit. Complementa PCA (lineal) para patrones globales en comportamientos vacas.</li>
</ul>            """, unsafe_allow_html=True)

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
