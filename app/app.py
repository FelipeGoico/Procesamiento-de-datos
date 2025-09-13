import base64
import streamlit as st
from pathlib import Path
import pandas as pd
from data_loader import get_full_data, get_sample_data, get_graph, prepare_data, init_data, init_tsne, load_umap_data, set_global_config


# ===========================
# Configuración página
# ===========================

set_global_config()


# ===========================
# Carpeta base y ruta de la imagen
# ===========================
# sube desde pages -> app -> Procesamiento-de-datos
BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "img"
IMAGE_PATH_02 = IMG_DIR / "fvets-12-1630083-g002.jpg"
IMAGE_PATH_03 = IMG_DIR / "fvets-12-1630083-g003.jpg"
IMAGE_PATH_04 = IMG_DIR / "fvets-12-1630083-g004.jpg"
IMAGE_PATH_05 = IMG_DIR / "fvets-12-1630083-g005.jpg"


# ===========================
# Convertir imagen a base64
# ===========================


def img_to_base64(img_path):
    with open(str(img_path), "rb") as f:  # <-- Convertimos Path a string
        data = f.read()
    return base64.b64encode(data).decode()


img_base64 = img_to_base64(IMAGE_PATH_05)

# ===========================
# Banner con imagen de fondo
# ===========================
st.markdown(f"""
<div style="
    background-image: url('data:image/jpg;base64,{img_base64}');
    background-size: cover;
    background-position: center;
    padding: 40px 20px; 
    border-radius: 10px; 
    text-align: center;
    color: white;
    font-size: 36px;
    font-weight: bold;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.5);  /* mejora la legibilidad */
    box-shadow: 3px 3px 15px #888888;
    width: 100%;
">
 EXAMEN PROCESAMIENTO DE DATOS - MAGISTER DATA SCIENCE 
<p style="font-size:18px; font-weight:normal; margin-top:10px; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);">
A dataset for detecting walking, grazing, and resting behaviors in free-grazing cattle using IoT collar IMU signals
</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# 1. Introducción
# ===========================
st.markdown("""
<div style="
    background-color:#e0e0e0; 
    padding: 10px; 
    border-radius: 5px; 
    text-align:center;
    font-size:24px;
    font-weight:bold;
    color:#333333;
    margin-top:20px;
">
📊 1. Introducción
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: justify;">

Para esta ocasión, como equipo hemos decidido orientar nuestro examen hacia nuestro objetivo final del Magíster, que es desarrollar una metodología robusta de procesamiento y análisis de datos de comportamiento animal.  

El enfoque de este proyecto no se limita únicamente al análisis exploratorio de un dataset público, sino que busca sentar las bases conceptuales y prácticas para nuestra tesis, permitiéndonos integrar técnicas de preprocesamiento, generación de variables derivadas y visualización avanzada en un flujo coherente y reproducible.

El dataset escogido para llevar a cabo esta investigación proviene de un estudio reciente que registra comportamientos de vacas de pastoreo —caminata, pastoreo y reposo— mediante collares IoT equipados con sensores IMU (MPU9250 y BNO055), con validación a través de video PTZ. Este dataset ofrece una base sólida con 441 eventos etiquetados, más de 7 horas de grabación y más de 3 GB de datos crudos, lo que permite explorar diversas estrategias de análisis y preparación de datos para modelamiento.

Nuestro objetivo específico es reutilizar este dataset para:
1. Generar nuevas variables derivadas que capturen información relevante de los sensores.  
2. Aplicar técnicas de reducción de dimensionalidad para facilitar la visualización y comprensión de los datos.  
3. Construir una interfaz interactiva en Streamlit que permita explorar los patrones de comportamiento de manera intuitiva, y que pueda ser extendida en el futuro hacia sistemas de monitoreo con drones o integraciones más complejas.

Este enfoque permite que el proyecto sirva como un <b>puente entre los conceptos aprendidos durante el magíster y la aplicación práctica en un escenario real</b>, fortaleciendo la capacidad de implementar soluciones de Data Science de forma integral, desde la adquisición y limpieza de datos hasta la generación de insights accionables.
</div>
""", unsafe_allow_html=True)

# ===========================
# 2. Proyecto Original
# ===========================
st.markdown("---")
st.markdown("## 2. Proyecto Original: Descripción técnica")

# 2.1 Recolección de datos
st.markdown("### 2.1 Recolección de datos")
st.markdown("""
<div style="text-align: justify; margin-bottom: 10px;">
La recolección de datos se llevó a cabo en vacas lecheras de pastoreo utilizando collares IoT equipados con sensores IMU (Unidad de Medición Inercial). Estos collares fueron colocados alrededor del cuello de 10 vacas lecheras.
</div>
""", unsafe_allow_html=True)
st.image(str(IMAGE_PATH_03),
         caption="Vacas lecheras de pastoreo con collares IoT", use_container_width=True)
st.markdown("""
<div style="text-align: justify;">
<ul>
<li><b>Ubicación:</b> Granja Experimental Maquehue, Universidad de La Frontera, Temuco, Chile.</li>
<li><b>Sujetos:</b> 10 vacas lecheras con collares IoT, colocados en el cuello.</li>
<li><b>Sensores:</b>
    <ul>
        <li><b>MPU9250:</b> Sensor económico de 9 ejes que mide aceleración, velocidad angular (giros) y magnetismo en tres dimensiones. Permite registrar los movimientos y cambios de orientación del animal en tiempo real, capturando patrones de actividad como caminar, girar o acostarse. Proporciona datos crudos que pueden ser procesados con algoritmos propios para análisis detallado de comportamiento.</li>
        <li><b>BNO055:</b> Sensor avanzado de 9 ejes con fusión de datos integrada (acelerómetro, giroscopio y magnetómetro). Entrega directamente la <b>orientación absoluta del collar en el espacio</b> mediante quaterniones, facilitando la interpretación de posturas y movimientos complejos sin necesidad de cálculos adicionales.</li>
    </ul>
</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.image(str(IMAGE_PATH_02),
         caption="Localización de la Granja Experimental Maquehue, Temuco, Chile")

# 2.2 Objetivo del proyecto
st.markdown("### 2.2 Objetivo del proyecto original")
st.markdown("""
<div style="text-align: justify;">
El objetivo principal del proyecto era capturar y analizar de manera precisa los movimientos y posturas de vacas lecheras de pastoreo, con el fin de:  

- Detectar comportamientos específicos, como caminar, acostarse, giros bruscos o sacudidas de cola.  
- Generar un dataset confiable que permita estudiar la relación entre los movimientos de las vacas y factores externos, como la presencia de insectos o condiciones ambientales.  

Para lograr esto, se utilizaron dos sensores complementarios:  

1. **MPU9250:** Proporciona datos crudos de aceleración, giro y magnetismo en 3 ejes, permitiendo un análisis detallado de patrones de movimiento mediante algoritmos personalizados.  
2. **BNO055:** Entrega orientación absoluta del collar en el espacio mediante fusión de datos interna, facilitando la interpretación de posturas y movimientos complejos sin necesidad de cálculos adicionales.  

**Beneficio de usar ambos sensores:**  
- Permite validar los datos entre sensores, aumentando la confiabilidad del registro de comportamiento.  
- Balance entre **detalle y facilidad de análisis**: datos crudos del MPU9250 y orientación lista para usar del BNO055.  
- Mejora la sincronización con las cámaras PTZ, asegurando que los eventos registrados por los sensores coincidan con la observación visual.
</div>
""", unsafe_allow_html=True)

st.image(str(IMAGE_PATH_04),
         caption="Diagrama de la metodología propuesta: proyecto original", use_container_width=True)

# 2.3 Variables originales y procesamiento
st.markdown("### 2.3 Variables originales y procesamiento")
st.markdown("""
<div style="text-align: justify;">
Las señales registradas incluyen aceleración lineal, velocidad angular, magnetómetro y cuaterniones (solo BNO055). Variables medidas tanto en marco corporal (“body frame”) como mundial (“world frame”) permiten estimaciones más robustas.

Para transformar aceleraciones del marco corporal al mundial se usan matrices de rotación derivadas de las orientaciones dadas por los cuaterniones. Además, los datos crudos deben escalarse a unidades físicas mediante factores de escala según sensibilidad y rango del sensor.
</div>
""", unsafe_allow_html=True)

# ===========================
# Transición desde 2.3 al proyecto propio
# ===========================
st.markdown("---")

st.markdown("""
<div style="
    background-color:#f2f2f2; 
    padding: 8px; 
    border-radius: 5px; 
    text-align:center;
    font-size:22px;
    font-weight:bold;
    color:#333333;
    margin-top:20px;
">
🔍 Introducción a Nuestro Proyecto
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: justify; margin-top:10px;">
El proyecto original constituye una base sólida para comprender el movimiento y la actividad de vacas lecheras mediante sensores IoT. 
A partir de esta experiencia, nuestro trabajo amplía la búsqueda incorporando <b>nuevas columnas derivadas</b> que permiten detectar 
comportamientos anómalos en los animales. 

En particular, proponemos observar señales vinculadas a la presencia de <b>moscas en vacas de lomo y cuerno</b>, cuyos efectos suelen 
manifestarse en espasmos, movimientos bruscos o patrones irregulares de actividad. Estas nuevas métricas podrían convertirse en 
indicadores tempranos de una enfermedad asociada a la infestación por moscas.

La posibilidad de <b>automatizar esta detección en la ganadería</b> representa un aporte clave: no solo permitiría alertar de manera 
temprana sobre problemas sanitarios, sino también optimizar el manejo productivo y reducir pérdidas asociadas al bienestar animal.
</div>
""", unsafe_allow_html=True)


# ===========================
# Variables Derivadas
# ===========================
st.subheader("Variables Derivadas")

st.markdown(
    """
<div style="text-align: justify;">
Además de las variables originales registradas por los sensores, se calcularon variables derivadas
con el fin de capturar patrones de comportamiento más informativos y robustos para los modelos de
reducción de dimensionalidad y clasificación.  

Estas nuevas columnas no solo permiten mejorar la separabilidad de actividades como caminar, pastorear
o descansar, sino que también abren la puerta a la detección de <b>comportamientos anómalos</b> relacionados
con la presencia de moscas en vacas de lomo y cuerno. Movimientos bruscos, espasmos o irregularidades 
en la magnitud de aceleración y giroscopio podrían constituir <b>indicadores tempranos</b> de infestaciones
o estrés animal.  

Contar con estas métricas derivadas representa una base sólida para escalar hacia un sistema de 
<b>automatización ganadera</b>, donde anomalías detectadas por sensores puedan generar alertas inmediatas, 
reduciendo riesgos y pérdidas productivas.
</div>
    """, unsafe_allow_html=True
)

st.markdown(
    """
**Magnitudes y Estadísticas de Movimiento**
- Magnitud de aceleración (`*_acc_magnitude`): Norma euclidiana del vector de aceleración √(AX² + AY² + AZ²).
- Magnitud del giroscopio (`*_gyro_magnitude`): Norma euclidiana del vector del giroscopio √(GX² + GY² + GZ²).
- Media móvil y desviación estándar (`acc_mean_window5`, `acc_std_window5`, `gyro_mean_window5`, `gyro_std_window5`) en ventana de 5 muestras (~0.5s).

**Indicadores Comportamentales**
- Inquietud alta (`inquietud_alta`): Categórica, indica periodos de agitación.
- Actividad extrema (`actividad_extrema`): Categórica, marca eventos de actividad significativamente superior al promedio.

**Justificación**
1. Reducir dimensionalidad inicial de sensores múltiples a métricas representativas.
2. Capturar dinámicas temporales mediante estadísticas en ventana deslizante.
3. Facilitar clasificación y visualización manteniendo interpretabilidad.
"""
)

st.markdown("---")

# ===========================
# Metodología
# ===========================
st.subheader("Metodología")

st.markdown(
    """
<div style="text-align: justify;">
El flujo metodológico propuesto integra desde la adquisición de datos hasta la reducción de dimensionalidad 
y visualización interactiva. Nuestra aproximación busca no solo validar la base del proyecto original, sino 
también extenderla hacia la detección temprana de anomalías.  

Este enfoque resulta clave para detectar signos sutiles de incomodidad o enfermedad en el ganado, como los 
provocados por infestaciones de moscas. Al derivar nuevas columnas que reflejen espasmos, agitaciones o 
cambios bruscos de orientación, generamos insumos concretos para una futura automatización en sistemas 
de monitoreo ganadero.
</div>
    """, unsafe_allow_html=True
)

st.markdown(
    """
**Preprocesamiento**
1. Carga y unificación de datos: lectura de CSV por carpetas de comportamiento.
2. Imputación de valores faltantes: KNNImputer (k=5) para preservar correlaciones locales.
3. Escalado de variables: StandardScaler (media=0, std=1) para PCA/LDA/t-SNE/UMAP.
4. Codificación de etiquetas y división Train/Test: LabelEncoder, 80/20 estratificado.

**Reducción de Dimensionalidad**
- PCA: Componentes principales lineales, scree plot para varianza acumulada.
- LDA: Supervisado, maximiza separabilidad entre clases.
- t-SNE: No lineal, preserva relaciones locales (perplexity=30, n_iter=1000).
- UMAP: No lineal, conserva estructura local/global (n_neighbors=15, min_dist=0.1).

**Evaluación**
- Supervisada: Accuracy con kNN (k=5) sobre embeddings 2D.
- No supervisada: Silhouette score con KMeans (k=4).
- Opcional: ARI y NMI con etiquetas verdaderas.

**Visualización e Interactividad**
- Scatterplots 2D de proyecciones, coloreados por comportamiento.
- Panel interactivo en Streamlit para explorar embeddings y ajustar parámetros dinámicamente.
"""
)

st.markdown("---")

# ===========================
# EDA (Exploratory Data Analysis)
# ===========================
st.subheader("Análisis Exploratorio de Datos (EDA)")

st.markdown(
    """
<div style="text-align: justify;">
A continuación, se presentan las visualizaciones y análisis exploratorios que permiten comprender 
la distribución de las variables, la relación entre ellas y los patrones de comportamiento registrados.  

Este paso constituye un puente entre la definición metodológica y la aplicación práctica, 
donde se pueden identificar tanto regularidades como comportamientos anómalos que respaldan 
la relevancia de nuestras columnas derivadas.
</div>
    """, unsafe_allow_html=True
)
# ===========================
# Botón de navegación al EDA
# ===========================
st.divider()
if st.button("🔍 Ir al Análisis Exploratorio (EDA)"):
    st.switch_page("pages/1_EDA.py")
if st.button("🔍 Ir al Análisis Discriminante Lineal (LDA)"):
    st.switch_page("pages/2_LDA.py")
if st.button("🔍 Ir al Análisis de Componentes Principales (PCA)"):
    st.switch_page("pages/3_PCA.py")
if st.button("🔍 Ir al t-Distributed Stochastic Neighbor Embedding (t-SNE)"):
    st.switch_page("pages/4_t_SNE.py")
if st.button("🔍 Ir al Uniform Manifold Approximation and Projection (UMAP)"):
    st.switch_page("pages/5_UMAP.py")


# Inicializar datos y gráficos
init_data()
init_tsne()
load_umap_data()
# Cargar dataset
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("../data_processed.csv")

# Pre-cargar gráficos apenas se inicia la app
if "graphs" not in st.session_state:
    st.session_state.graphs = get_graph(st.session_state.df)
fig, col_numericas = st.session_state.graphs

# Convertir el dataframe a CSV (en memoria, sin index)
csv = get_full_data().to_csv(index=False)

# Botón de descarga
st.download_button(
    label="📥 Descargar Datos CSV",
    data=csv,
    file_name="datos.csv",
    mime="text/csv",
)
