from app import st

st.set_page_config(page_title="LDA",
                   page_icon=":bar_chart:", layout="wide")
st.title("Análisis Discriminante Lineal (LDA)")
st.write("""
El Análisis Discriminante Lineal (LDA) es una técnica supervisada utilizada para
reducir la dimensionalidad de un conjunto de datos mientras se maximiza la separabilidad
entre las clases. A diferencia del PCA, que es una técnica no supervisada, el LDA tiene en
cuenta las etiquetas de clase al buscar las combinaciones lineales de características que
mejor separan las diferentes clases. Esto lo hace especialmente útil para problemas de clasificación.
""")
st.write("""
El LDA funciona proyectando los datos en un espacio de menor dimensión, donde las clases
están lo más separadas posible. Esto se logra mediante la maximización de la razón de
varianza entre clases a la varianza dentro de las clases. El resultado es un conjunto de
nuevas características (componentes discriminantes) que pueden ser utilizadas para entrenar
modelos de clasificación más efectivos.
""")
st.write("""
El LDA es ampliamente utilizado en diversas aplicaciones, como el reconocimiento de patrones,
la visión por computadora y la bioinformática, donde la reducción de dimensionalidad y la
clasificación precisa son cruciales.
""")
