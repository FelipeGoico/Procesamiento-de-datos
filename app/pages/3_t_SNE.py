from app import st

st.set_page_config(page_title="t-SNE",
                   page_icon=":bar_chart:", layout="wide")
st.title("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
st.write("""
t-Distributed Stochastic Neighbor Embedding (t-SNE) es una técnica de reducción de
dimensionalidad no supervisada que se utiliza principalmente para la visualización de datos
de alta dimensión en espacios de menor dimensión (generalmente 2D o 3D). Fue desarrollado por
Laurens van der Maaten y Geoffrey Hinton en 2008.
""")
st.write("""
El objetivo principal de t-SNE es preservar las relaciones de proximidad entre los puntos de
datos en el espacio de alta dimensión al mapearlos a un espacio de menor dimensión. A
diferencia de otras técnicas de reducción de dimensionalidad, como PCA o LDA, t-SNE se
enfoca en mantener las distancias locales entre los puntos de datos, lo que lo hace
particularmente útil para visualizar estructuras complejas y agrupamientos en los datos.
""")
st.write("""
El algoritmo t-SNE funciona en dos etapas principales:
1. Cálculo de las probabilidades de similitud en el espacio de alta dimensión:
   - Para cada par de puntos de datos, se calcula la probabilidad de que un punto
     elija a otro como su vecino, basándose en una distribución gaussiana centrada
     en el primer punto.
   - Esto da lugar a una matriz de probabilidades que refleja las relaciones de
     proximidad en el espacio original.
2. Mapeo a un espacio de menor dimensión:
   - Se inicializan aleatoriamente las posiciones de los puntos en el espacio de
     menor dimensión.
   - Se define una distribución t de Student para calcular las probabilidades de
     similitud en el espacio de menor dimensión.
   - El algoritmo minimiza la divergencia de Kullback-Leibler entre las dos
     distribuciones de probabilidad (la del espacio original y la del espacio
     reducido) utilizando un método de optimización, como el descenso de
     gradiente.
""")
st.write("""
t-SNE es ampliamente utilizado en diversas aplicaciones, como la visualización de datos
genómicos, el análisis de imágenes, el procesamiento del lenguaje natural y la exploración de
datos en general. Sin embargo, es importante tener en cuenta que t-SNE puede ser computacionalmente
intensivo y sensible a los hiperparámetros, como la perplexidad y la tasa de aprendizaje, lo que
requiere una cuidadosa selección y ajuste para obtener resultados óptimos.
""")
st.write("""
En resumen, t-SNE es una poderosa herramienta para la visualización de datos de alta dimensión
que ayuda a revelar estructuras y patrones ocultos en los datos, facilitando su interpretación
y análisis.
""")
st.write("""Referencias:
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
""")
st.write("""
- https://distill.pub/2016/misread-tsne/
""")
st.write("""- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
""")
st.write("""- https://lvdmaaten.github.io/tsne/
""")
st.write("""- https://www.youtube.com/watch?v=NEaUSP4YerM
""")
