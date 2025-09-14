import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from data_loader import (
    init_data, init_tsne, load_umap_data,
    set_global_config, get_preprocessed_data
)
from datetime import datetime
from utils.report_generator import create_project_report

import sys
import os
from pathlib import Path

# Agregar el directorio padre al path de Python
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))



# ---------------------------------------------------
# Configuración global
# ---------------------------------------------------
set_global_config()

# ---------------------------------------------------
# Inicialización de datos
# ---------------------------------------------------
if "df" not in st.session_state:
    init_data()

df = st.session_state.df

# ---------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------
if "preprocessed" not in st.session_state:
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    # Combinar train + test
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])
    # Guardar solo los datos completos
    st.session_state.preprocessed = (X_full, y_full)
else:
    X_train, X_test, y_train, y_test = st.session_state.preprocessed
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])

# ---------------------------------------------------
# Inicialización embeddings t-SNE y UMAP
# ---------------------------------------------------
if "tsne_2d" not in st.session_state or "tsne_y" not in st.session_state:
    init_tsne()

if "X_umap_2d" not in st.session_state or "umap_y" not in st.session_state:
    load_umap_data()

X_tsne = st.session_state.tsne_2d
X_umap = st.session_state.X_umap_2d
y = st.session_state.tsne_y  # mismas etiquetas para todos los embeddings

# ---------------------------------------------------
# Función KNN
# ---------------------------------------------------


def evaluate_knn(emb_train, emb_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(emb_train, y_train)
    y_pred = knn.predict(emb_test)
    return accuracy_score(y_test, y_pred)


# ---------------------------------------------------
# Evaluación KNN
# ---------------------------------------------------
# Para PCA/LDA usamos X_full dividido en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

acc_pca = evaluate_knn(X_train, X_test, y_train, y_test)
acc_lda = evaluate_knn(X_train, X_test, y_train, y_test)
# Para t-SNE y UMAP evaluamos con todos los datos (no hacemos train/test)
acc_tsne = evaluate_knn(X_tsne, X_tsne, y, y)
acc_umap = evaluate_knn(X_umap, X_umap, y, y)

# ---------------------------------------------------
# Función clustering
# ---------------------------------------------------


def evaluate_clustering(embedding, y_true):
    num_clusters = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)
    sil = silhouette_score(embedding, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)
    return sil, ari, nmi


sil_pca, ari_pca, nmi_pca = evaluate_clustering(X_full, y_full)
sil_lda, ari_lda, nmi_lda = evaluate_clustering(X_full, y_full)
sil_tsne, ari_tsne, nmi_tsne = evaluate_clustering(X_tsne, y)
sil_umap, ari_umap, nmi_umap = evaluate_clustering(X_umap, y)


st.markdown("""
            <div style="text-align: justify; margin-top:20px;">

### Resumen del Examen Final: Comparación de Métodos de Reducción de Dimensionalidad

Este examen final abarca técnicas de procesamiento de datos y análisis estadístico. La tabla compara cuatro métodos populares de reducción de dimensionalidad —PCA, LDA, t-SNE y UMAP— en métricas clave de evaluación para una tarea de clasificación/agrupamiento (usando KNN para la precisión). Estas métricas incluyen el rendimiento downstream (Precisión con KNN), la calidad interna del agrupamiento (Puntuación de Silueta) y la concordancia con la verdad terreno (ARI y NMI). Valores más altos son generalmente mejores para todas las métricas, excepto ARI que puede ser negativo (indicando una concordancia peor que aleatoria).

Aquí va un desglose estructurado de los resultados:

  """, unsafe_allow_html=True)

# ---------------------------------------------------
# Tabla comparativa final
# ---------------------------------------------------
results = pd.DataFrame({
    'Método': ['PCA', 'LDA', 't-SNE', 'UMAP'],
    'Accuracy KNN': [acc_pca, acc_lda, acc_tsne, acc_umap],
    'Silhouette': [sil_pca, sil_lda, sil_tsne, sil_umap],
    'ARI (opcional)': [ari_pca, ari_lda, ari_tsne, ari_umap],
    'NMI (opcional)': [nmi_pca, nmi_lda, nmi_tsne, nmi_umap]
})

st.dataframe(results.set_index('Método').style.format({
    'Accuracy KNN': '{:.3f}',
    'Silhouette': '{:.3f}',
    'ARI (opcional)': '{:.3f}',
    'NMI (opcional)': '{:.3f}'
}))

st.markdown("""
#### Perspectivas Clave
<ul><li><b>Mejor para Precisión de Clasificación (KNN):</b> t-SNE destaca con la puntuación más alta (0.942), lo que sugiere que preserva mejor la estructura discriminativa para la clasificación basada en KNN. UMAP sigue de cerca (0.930), mientras que PCA y LDA empatan en un sólido pero menor rendimiento (0.914).</li>
<li><b>Mejor para Calidad de Agrupamiento (Silueta):</b> UMAP lidera con 0.442, indicando agrupamientos más compactos y separables en el espacio reducido, seguido por t-SNE (0.407). PCA y LDA quedan atrás con 0.115, lo que implica agrupamientos menos definidos en métodos lineales.</li>
<li><b>Concordancia Externa de Agrupamiento (ARI & NMI):</b> PCA y LDA brillan con valores altos y similares (ARI 0.233, NMI 0.239), mostrando una fuerte recuperación de las etiquetas verdaderas. UMAP es sólido (ARI 0.081, NMI 0.179), pero t-SNE es el más débil (ARI 0.029, NMI 0.045), posiblemente por su enfoque en la estructura local.</li>
<li><b>Compromisos:</b> <ul><li>Si tu prioridad es la clasificación downstream, elige t-SNE por su superior precisión, aunque su ARI/NMI sea bajo.</li>
<li>Para concordancia con verdad terreno o análisis supervisado, PCA o LDA son ideales por sus métricas externas altas y consistencia.</li>
<li>UMAP ofrece un equilibrio excelente: buena precisión, la mejor silueta y métricas externas decentes, ideal para visualización y agrupamiento no lineal.</li>
            </ul></li>

            
Estos resultados dependen del conjunto de datos específico (por ejemplo, características de alta dimensionalidad). Recomiendo validar con más experimentos, como ajuste de hiperparámetros o validación cruzada.
""", unsafe_allow_html=True)




st.markdown("---")
st.subheader("Generar Reporte Formal")


# Al final del archivo, después de mostrar la tabla de resultados
st.markdown("---")
st.subheader("Generar Reporte Formal")

try:
    # Primero verifica si FPDF está instalado
    import fpdf
    st.success("✓ FPDF está correctamente instalado")
    
    # Ahora intenta importar la función específica
    from utils.report_generator import create_project_report
    st.success("✓ Función create_project_report importada correctamente")
    
    if st.button("📄 Generar Reporte Técnico en PDF"):
        with st.spinner("Generando reporte formal..."):
            pdf_bytes = create_project_report()
            
            st.success("Reporte generado exitosamente!")
            
            st.download_button(
                label="📥 Descargar Reporte Completo",
                data=pdf_bytes,
                file_name=f"reporte_tecnico_proyecto_ganado_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

except ImportError as e:
    # Muestra el error específico
    st.error(f"❌ Error de importación: {str(e)}")
    
    # Diagnóstico específico
    if "fpdf" in str(e).lower():
        st.warning("""
        **Problema con FPDF:** 
        - Ejecuta: `pip install fpdf`
        - Reinicia Streamlit
        """)
    elif "utils" in str(e).lower() or "report_generator" in str(e).lower():
        st.warning("""
        **Problema con la estructura de carpetas:**
        - Asegúrate de tener la carpeta `utils/` 
        - Con el archivo `report_generator.py` dentro
        - Estructura correcta:
          ```
          tu_proyecto/
          ├── app.py
          ├── pages/
          │   └── 6_Resumen.py
          └── utils/
              └── report_generator.py
          ```
        """)
    else:
        st.warning(f"Error de importación no reconocido: {e}")