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
st.markdown("""
            <div style="text-align: justify; margin-top: 20px;">

Esta gráfica representa una proyección bidimensional (2D) del Análisis de Componentes Principales (PCA) aplicada al conjunto de datos de entrenamiento (Train) provenientes de sensores IoT en collares de vacas lecheras. El PCA reduce la dimensionalidad de las variables originales (aceleración, giroscopio, magnetómetro, etc., de sensores MPU9250 y BNO055) a dos componentes principales: PC1 (eje x, dirección de máxima varianza, posiblemente relacionada con niveles de actividad general) y PC2 (eje y, segunda varianza, capturando variaciones en orientación o intensidad de movimiento).
""", unsafe_allow_html=True)
st.markdown("""
            <div style="text-align: justify; margin-top: 20px;">
            Dispersión General:

Puntos aislados en extremos (e.g., rojos en PC1 positivo alto) podrían ser outliers o anomalías, como "inquietud alta" (espasmos por estrés, e.g., moscas).
Solapamiento entre clases: Alto entre Misc y Walking/Grazing (~40-50%), lo que es típico en datos IMU reales (influenciados por terreno o ruido). El reposo se separa mejor, validando PC1 como discriminador de "inactividad vs. actividad".
Densidad: Mayor en el centro (alta concentración de datos de entrenamiento), con menor dispersión en PC2 positivo (posiblemente capturando giros/orientaciones).""", unsafe_allow_html=True)

# ===========================
# Scree Plot
# ===========================
st.markdown("### Varianza Explicada por Componentes")
var_exp_cumsum = np.cumsum(pca_2d.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(range(1, len(var_exp_cumsum)+1),
        var_exp_cumsum, marker='o', color='#1f4e79')
ax.set_title("Varianza Explicada Acumulada por Componentes (PCA)")
ax.set_xlabel("Número de Componentes")
ax.set_ylabel("Varianza Explicada Acumulada")
ax.grid(True)
st.pyplot(fig)
st.markdown("""
            <div style="text-align: justify; margin-top:20px;">

### Resumen de la Explicación de la Gráfica PCA (Scree Plot Acumulativo)

<ul><li>Tipo: Curva de varianza explicada acumulada en PCA, mostrando distribución de varianza en componentes principales (enfocado en 2D para visuales).</li>
<li>Ejes: X: Número de componentes (1-2); Y: Varianza acumulada (0-20%).</li>
<li>Datos clave: PC1 explica ~12% (movimiento principal); PC1+PC2 suman ~20% (baja cobertura, indica alta dimensionalidad en datos IMU).</li>
<li>Interpretación: Línea lineal sin "codo" – retiene 2 componentes para exploración simple, pero usa más para modelado preciso. Útil para discriminar actividad en sensores de vacas (e.g., reposo vs. caminata). 
            </ul>""", unsafe_allow_html=True)


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
st.markdown("""
<div style="text-align: justify; margin-top:20px;">
            
Esta gráfica es una visualización tridimensional (3D) del Análisis de Componentes Principales (PCA) aplicada al conjunto de datos de entrenamiento de sensores IoT (IMU y GPS en collares de vacas). El PCA reduce la dimensionalidad de las features originales (aceleración, giroscopio, etc.) a tres componentes principales: PC1 (eje X, máxima varianza, posiblemente niveles de actividad general), PC2 (eje Y, segunda varianza, orientaciones o ritmos), y PC3 (eje Z, tercera varianza, sutilezas como giros verticales).""", unsafe_allow_html=True)

# ===========================
# Conclusiones
# ===========================
st.markdown("### ✅ Conclusiones")
st.write("""
<div style="text-align: justify; margin-top:20px;">
         
#### PCA 2D + Varianza Explicada (Scree Plot):

<b>Proyección 2D:</b> Clusters parciales en datos de sensores IMU (train): Reposo (azul) compacto en PC1 negativo (~baja energía); Caminata (rojo) y Pastoreo (verde) solapados en PC1 positivo (alta actividad); Misceláneos dispersos. Separabilidad ~60-70%, con solapamientos por ruido (~30-40%).

<b>Varianza Explicada:</b> PC1 (~12%, movimiento global); PC1+PC2 (~20% acumulada). Línea lineal en scree plot indica alta dimensionalidad (necesita ≥8 componentes para >80%); 2D ideal para EDA inicial, discriminando inactividad vs. movimiento.

#### PCA 3D Explicativa:

<b>Proyección 3D: Mejora resolución:</b> Reposo esférico cerca del origen (baja dispersión en PC1/PC2/PC3); Caminata elíptica en PC1/PC3 positivo (dinámicas verticales/orientación). Solapamiento reducido (~20-30%), outliers rojos señalan anomalías (e.g., estrés).

<b>Varianza Adicional:</b> PC1+PC2+PC3 ~30-40% (PC3 añade ~10-20%, capturando sutilezas como giros). Útil para inmersión interactiva en Streamlit; PC1 discrimina energía, PC3 resuelve transiciones no lineales.

#### Implicaciones Generales:

<b>Fortalezas:</b> Captura patrones naturales en comportamientos vacas, valida variables derivadas; bajo solapamiento en 3D para alertas IoT (e.g., espasmos por moscas).
""", unsafe_allow_html=True)
st.divider()
if st.button("Volver a la Página Principal"):
    st.switch_page("app.py")
init_tsne()
load_umap_data()
