import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from data_loader import get_preprocessed_data

# ===========================
# Página y banner
# ===========================
st.set_page_config(
    page_title="PCA - Dataset db-cow-walking-IoT",
    page_icon=":bar_chart:",
    layout="wide"
)

st.markdown("""
<div style="background-color:#1f4e79; padding: 18px; border-radius: 10px; text-align:center; color: white;">
    <h1 style="margin:0;">📊 Análisis de Componentes Principales (PCA)</h1>
    <p style="margin:0; font-size:18px;">Reducción de dimensionalidad y exploración de patrones de comportamiento</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# Introducción
# ===========================
st.markdown("""
<div style="text-align: justify; margin-top: 20px;">
El Análisis de Componentes Principales (PCA) es una técnica estadística para reducir
la dimensionalidad de un conjunto de datos mientras se conserva la mayor cantidad
posible de varianza.  
En este proyecto aplicamos PCA a los datos de sensores de vacas (IMU y GPS) para:
<ul>
<li>Visualizar patrones de comportamiento.</li>
<li>Explorar agrupamientos naturales entre clases de actividad.</li>
<li>Servir como base para técnicas de clasificación supervisada.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ===========================
# Datos preprocesados
# ===========================
df = st.session_state.df
X_train, X_test, y_train, y_test = get_preprocessed_data()

feature_cols = [col for col in df.columns if col != 'label' and df[col].dtype in [np.float64, np.int64]]
X = df[feature_cols]
y = df['label']

# ---------------------------
# Train/Test, imputación y escalado
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

imputer = KNNImputer(n_neighbors=5)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=feature_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_cols)

# ===========================
# PCA 2D
# ===========================
st.markdown("### PCA 2D")
pca_2d = PCA(n_components=2)
X_train_pca2 = pca_2d.fit_transform(X_train_scaled)
df_pca2 = pd.DataFrame(X_train_pca2, columns=['PC1','PC2'])
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
Se observan agrupamientos que reflejan similitud de comportamiento entre las clases.  
Esto permite identificar relaciones y posibles solapamientos entre actividades.
""")

# ===========================
# Scree Plot
# ===========================
st.markdown("### Varianza Explicada por Componentes")
var_exp_cumsum = np.cumsum(pca_2d.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(range(1, len(var_exp_cumsum)+1), var_exp_cumsum, marker='o', color='#1f4e79')
ax.set_title("Varianza Explicada Acumulada por Componentes (PCA)")
ax.set_xlabel("Número de Componentes")
ax.set_ylabel("Varianza Explicada Acumulada")
ax.grid(True)
st.pyplot(fig)
st.write(f"PC1 explica {pca_2d.explained_variance_ratio_[0]:.2f} de la varianza, PC2 {pca_2d.explained_variance_ratio_[1]:.2f}.")

# ===========================
# PCA 3D
# ===========================
st.markdown("### PCA 3D Interactivo")
pca_3d = PCA(n_components=3)
X_train_pca3 = pca_3d.fit_transform(X_train_scaled)
df_pca3 = pd.DataFrame(X_train_pca3, columns=['PC1','PC2','PC3'])
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
La visualización 3D permite explorar más dimensiones de los datos y mejorar la
identificación de agrupamientos y separaciones entre clases, especialmente si
existen solapamientos en 2D.
""")

# ===========================
# Conclusiones
# ===========================
st.markdown("## ✅ Conclusiones")
st.write("""
- PCA permite reducir dimensionalidad y visualizar patrones generales de los datos.
- En 2D se observan agrupamientos parciales por comportamiento, útiles para análisis exploratorio.
- La representación 3D ofrece mayor claridad para identificar separaciones entre clases.
- Estos resultados sirven como base para aplicar técnicas de clasificación supervisada
  (LDA, Random Forest, etc.) y para la detección de anomalías en comportamientos.
""")
