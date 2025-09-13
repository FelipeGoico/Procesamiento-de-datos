import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_loader import init_data, init_tsne, load_umap_data, get_preprocessed_data, set_global_config

set_global_config()

# Inicializar datos y gráficos
init_data()


# ===========================
# Datos ya preprocesados
# ===========================

if "preprocessed" not in st.session_state:

    X_train, X_test, y_train, y_test = get_preprocessed_data()
    st.session_state.preprocessed = (
        X_train, X_test, y_train, y_test)
else:
    X_train, X_test, y_train, y_test = st.session_state.preprocessed

# ===========================
# Configuración de página
# ===========================
st.set_page_config(
    page_title="PCA - Dataset db-cow-walking-IoT",
    page_icon=":bar_chart:",
    layout="wide"
)

# ===========================
# Banner
# ===========================
st.markdown("""
<div style="background-color:#1f4e79; padding:18px; border-radius:10px; text-align:center; color:white;">
    <h1 style="margin:0;">📊 Análisis de Componentes Principales (PCA)</h1>
    <p style="margin:0; font-size:18px;">Reducción de dimensionalidad y exploración de patrones de comportamiento</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# Introducción
# ===========================
st.markdown("""
<div style="text-align: justify; margin-top:20px;">
El PCA reduce la dimensionalidad conservando varianza. Aplicamos PCA a los datos de sensores de vacas (IMU y GPS) para:
<ul>
<li>Visualizar patrones de comportamiento.</li>
<li>Explorar agrupamientos naturales entre clases.</li>
<li>Servir como base para clasificación supervisada.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ===========================
# PCA 2D
# ===========================
st.markdown("### PCA 2D")
pca_2d = PCA(n_components=2)
X_train_pca2 = pca_2d.fit_transform(X_train)
df_pca2 = pd.DataFrame(X_train_pca2, columns=['PC1', 'PC2'])
df_pca2['label'] = y_train.values

fig_2d = px.scatter(
    df_pca2, x='PC1', y='PC2', color='label',
    opacity=0.7, width=800, height=500,
    title="Proyección PCA 2D de los datos de sensores (Train)"
)
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("#### Interpretación PCA 2D")
st.write("""
Cada punto representa una muestra proyectada sobre las dos componentes principales.  
Se observan agrupamientos que reflejan similitud de comportamiento entre clases.
""")

# ===========================
# Scree Plot
# ===========================
st.markdown("### Varianza Explicada por Componentes")
var_exp_cumsum = np.cumsum(pca_2d.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(var_exp_cumsum)+1),
        var_exp_cumsum, marker='o', color='#1f4e79')
ax.set_title("Varianza Explicada Acumulada por Componentes (PCA)")
ax.set_xlabel("Número de Componentes")
ax.set_ylabel("Varianza Explicada Acumulada")
ax.grid(True)
st.pyplot(fig)
st.write(
    f"PC1 explica {pca_2d.explained_variance_ratio_[0]:.2f} de la varianza, PC2 {pca_2d.explained_variance_ratio_[1]:.2f}.")

# ===========================
# PCA 3D
# ===========================
st.markdown("### PCA 3D Interactivo")
pca_3d = PCA(n_components=3)
X_train_pca3 = pca_3d.fit_transform(X_train)
df_pca3 = pd.DataFrame(X_train_pca3, columns=['PC1', 'PC2', 'PC3'])
df_pca3['label'] = y_train.values

fig_3d = px.scatter_3d(
    df_pca3, x='PC1', y='PC2', z='PC3',
    color='label', opacity=0.7,
    width=800, height=600,
    title="Proyección PCA 3D interactiva (Train)"
)
fig_3d.update_traces(marker=dict(size=4))
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("#### Interpretación PCA 3D")
st.write("""
La visualización 3D permite explorar más dimensiones de los datos y mejorar la identificación de agrupamientos y separaciones entre clases.
""")

# ===========================
# Conclusiones
# ===========================
st.markdown("## ✅ Conclusiones")
st.write("""
- PCA permite reducir dimensionalidad y visualizar patrones generales.
- En 2D se observan agrupamientos parciales por comportamiento.
- La representación 3D ofrece mayor claridad para identificar separaciones entre clases.
- Estos resultados sirven como base para aplicar clasificación supervisada y detección de anomalías.
""")
st.divider()
if st.button("Volver a la Página Principal"):
    st.switch_page("app.py")
init_tsne()
load_umap_data()
