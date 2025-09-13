import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data_loader import get_preprocessed_data, init_data, init_tsne, load_umap_data, set_global_config

set_global_config()
init_data()
# ===========================
# Configuración de la página
# ===========================
# st.set_page_config(
#     page_title="LDA - Dataset db-cow-walking-IoT",
#     page_icon=":bar_chart:",
#     layout="wide"
# )

# ===========================
# Banner
# ===========================
st.markdown("""
<div style="background-color:#1f4e79; padding: 18px; border-radius: 10px; text-align:center; color: white;">
    <h1 style="margin:0;">🧩 Análisis Discriminante Lineal (LDA)</h1>
    <p style="margin:0; font-size:18px;">Reducción supervisada de dimensionalidad y clasificación de comportamientos</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# Introducción
# ===========================
st.markdown("""
<div style="text-align: justify; margin-top: 20px;">
El <b>Análisis Discriminante Lineal (LDA)</b> es una técnica supervisada utilizada para reducir la dimensionalidad 
de un conjunto de datos mientras se maximiza la separabilidad entre clases.  
A diferencia del PCA, que es no supervisado, el LDA utiliza las etiquetas de clase para encontrar las combinaciones lineales de características que mejor separan las clases.  

Esto se logra proyectando los datos en un espacio de menor dimensión donde las clases están lo más separadas posible, 
maximizando la razón de varianza entre clases frente a la varianza dentro de las clases.  
El resultado son nuevas características (componentes discriminantes) útiles para entrenar modelos de clasificación más precisos.
</div>
""", unsafe_allow_html=True)

# ===========================
# Datos ya preprocesados
# ===========================


if "preprocessed" not in st.session_state:
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    st.session_state.preprocessed = (X_train, X_test, y_train, y_test)
else:
    X_train, X_test, y_train, y_test = st.session_state.preprocessed

# ===========================
# LDA 2D
# ===========================
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

df_lda = pd.DataFrame(X_train_lda, columns=['LD1', 'LD2'])
df_lda['label'] = y_train

fig_lda = px.scatter(
    df_lda, x='LD1', y='LD2', color='label',
    opacity=0.8, width=800, height=500,
    title="Proyección LDA 2D de los datos de sensores (Train)"
)
st.plotly_chart(fig_lda, use_container_width=True)

# ===========================
# Explicación de componentes
# ===========================
st.markdown("### 🔹 Explicación de componentes")
st.write(
    f"Explained variance ratio (aprox. discriminación entre clases): {lda.explained_variance_ratio_}"
)
st.divider()
if st.button("Volver a la Página Principal"):
    st.switch_page("app.py")
init_tsne()
load_umap_data()
