import streamlit as st
from pathlib import Path

# Carpeta base y ruta de la imagen
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # sube desde pages -> app -> Procesamiento-de-datos
IMG_DIR = BASE_DIR / "img"
IMAGE_PATH = IMG_DIR / "fvets-12-1630083-g002.jpg"

# Verificación de existencia de la imagen
if IMAGE_PATH.exists():
    st.image(str(IMAGE_PATH), caption="Granja Experimental Maquehue, Temuco, Chile, lugar donde lleve a cabo el estudio presente")
else:
    st.error(f"No se encontró la imagen en la ruta: {IMAGE_PATH}")

st.set_page_config(page_title="UMAP", page_icon=":bar_chart:", layout="wide")
st.title("Uniform Manifold Approximation and Projection (UMAP)")
st.write("This is a placeholder for UMAP content.")

st.markdown("# Análisis Exploratorio & Proyección: db-cow-walking-IoT")

st.markdown("## 1. Introducción")
st.markdown("""
Para esta ocasión, como equipo hemos decidido orientar nuestro examen hacia nuestro objetivo final (TESIS).  
El dataset escogido para llevar a cabo esta investigación ha sido un estudio público reciente que registra comportamientos de vacas de pastoreo — caminata, pastoreo y reposo — mediante collares IoT con sensores IMU (MPU9250 y BNO055) y validación por video PTZ.  
El paper original aporta un dataset estructurado con 441 eventos etiquetados, más de 7 h de grabación, y más de 3 GB de datos crudos, lo que lo convierte en un punto de partida sólido.

Nuestro objetivo específico es reutilizar este dataset para:  
1. Generar nuevas variables derivadas.  
2. Aplicar reducción de dimensionalidad.  
3. Crear una interfaz interactiva en Streamlit que posibilite análisis exploratorio y visualización avanzada, con visión futura hacia integración con drones para monitoreo territorial.
""")

st.markdown("---")
st.markdown("## 2. Proyecto Original: Descripción técnica")
st.markdown("### 2.1 Recolección de datos")
st.markdown("""
La recolección de datos se llevó a cabo en vacas lecheras de pastoreo utilizando collares IoT equipados con sensores IMU (Unidad de Medición Inercial). Estos collares fueron colocados alrededor del cuello de 10 vacas lecheras.  
([frontiersin.org](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2025.1630083/full?utm_source=chatgpt.com))

- **Ubicación:** Granja Experimental Maquehue, Universidad de La Frontera, Temuco, Chile.  
- **Sujetos:** 10 vacas lecheras con collares IoT, colocados en el cuello.  
- **Sensores:**
  - **MPU9250:** IMU de 9 ejes, económico, registra aceleración, giroscopio y magnetómetro.  
  - **BNO055:** IMU fusionado, permite medir orientación como cuaterniones para estimar marco mundial/world frame.  
- **Frecuencia de muestreo:** 10 Hz.  
- **Validación visual:** cámaras PTZ con visión nocturna, sincronizadas con las señales de los sensores.  
- **Estructura de almacenamiento:** archivos CSV por evento (nombre codificado: evento, comportamiento, identificador de vaca, fecha–hora), organizados jerárquicamente por tipo de comportamiento.  
""")

# Imagen
st.image(str(IMG_DIR / "fvets-12-1630083-g002.jpg"), caption="Localización de la Granja Experimental Maquehue, Temuco, Chile")

st.markdown("### 2.2 Variables originales y procesamiento")
st.markdown("""
Las señales registradas incluyen aceleración lineal, velocidad angular, magnetómetro, cuaterniones (solo BNO055) y GPS. Variables medidas tanto en marco corporal (“body frame”) como mundial (“world frame”) permiten estimaciones más robustas.

Para transformar aceleraciones del marco corporal al mundial se usan matrices de rotación derivadas de las orientaciones dadas por los cuaterniones. Además, los datos crudos deben escalarse a unidades físicas mediante factores de escala según sensibilidad y rango del sensor.
""")

st.markdown("---")
st.markdown("## 4. Variables derivadas & Reducción de Dimensionalidad")
st.markdown("""
Basándonos en los datos originales, hemos generado variables adicionales útiles para análisis y modelado:

- **Magnitudes de aceleración y giroscopio:**  
""")
st.latex(r"\text{acc\_magnitude} = \sqrt{AX^2 + AY^2 + AZ^2}")
st.latex(r"\text{gyro\_magnitude} = \sqrt{GX^2 + GY^2 + GZ^2}")

st.markdown("""
- Estadísticos móviles (ventanas deslizantes, por ejemplo de 5 muestras) para media y desviación estándar de aceleración y giro.  
- Indicadores binarios: “actividad extrema”, “inquietud”, etc., usando umbrales basados en la variabilidad de señales.

Para la reducción de dimensionalidad, consideraremos técnicas como **PCA (Análisis de Componentes Principales)**, **Selección de características** y/o técnicas de embedding (t-SNE, UMAP) para visualización de relaciones entre eventos.
""")

st.markdown("---")
st.markdown("## 5. Metodología propuesta para mi proyecto")
st.markdown("""
1. **Preprocesamiento:** limpiar señales, sincronizar timestamps, escalar valores, detectar y manejar outliers.  
2. **Generación de variables derivadas**, y creación de dataset estructurado.  
3. **División de datos para entrenamiento / validación**, experimentos con diferentes algoritmos supervisados: SVM, Random Forest, etc.  
4. **Reducción de dimensionalidad y visualización en Streamlit**, permitiendo explorar comportamientos por vaca, por evento, y visualizar componentes principales.  
5. **Proyección futura hacia drones:** usar sensores locales + cámara aérea + GPS para ampliar cobertura, capturar variables ambientales y sincronizar con los datos del collar, para detectar patrones más complejos.
""")

st.markdown("---")
st.markdown("## 6. Conclusiones")
st.markdown("""
El dataset original proporciona una base sólida: gran riqueza en señales IMU, etiquetado de comportamientos, estructura clara y buenos ejemplos de uso en clasificación. Sin embargo, mi proyecto se diferencia al incorporar variables derivadas, visualizar de forma interactiva, y preparar el camino hacia integración con sistemas aéreos (drones) para monitoreo territorial.

Se espera que, al aplicar reducción de dimensionalidad y crear una app en Streamlit, los resultados sean fácilmente interpretables por distintos actores (investigadores, veterinarios, productores), además de escalables.
""")
