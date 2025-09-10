from app import st, df
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="EDA",
                   page_icon=":bar_chart:", layout="wide")

multi = """
# Examen: Reducción de Dimensionalidad y Aplicación Interactiva con Streamlit

# Análisis Exploratorio de Datos (EDA) del Dataset db-cow-walking-IoT

Entender el problema que se intenta resolver. Esto incluye conocer el origen de los datos, su significado, y qué tipo de información se espera obtener.

## Introducción

El dataset "db-cow-walking-IoT" contiene datos de sensores inerciales (IMU) y GPS recolectados de vacas lecheras en un entorno de pastoreo.

El objetivo principal es clasificar y analizar los comportamientos de las vacas, como caminar, pastorear y descansar, utilizando técnicas de aprendizaje automático. 

Este análisis exploratorio de datos (EDA) proporciona una visión general del dataset, sus características principales, y estadísticas descriptivas relevantes.


## 1. Descripción General del Dataset

Fuente y Recolección: Los datos fueron recolectados en la Granja Experimental Maquehue, Temuco, Chile, entre mayo y octubre de 2024.
Se utilizaron collares IoT en 10 vacas lecheras, con dos tipos de IMU: MPU9250 y BNO055.
El dataset contiene 441 eventos etiquetados, representando más de 7 horas y 34 minutos de grabación.
- Una IMU (Unidad de Medición Inercial) es un conjunto de sensores que mide la aceleración lineal, la velocidad angular y la orientación de un objeto, típicamente usando un acelerómetro, un giroscopio y un magnetómetro.
- El MPU9250 es un sensor IMU de 9 ejes que combina un acelerómetro, un giroscopio y un magnetómetro en un solo chip.
- El BNO055 es un sensor IMU de 9 ejes con un microcontrolador integrado que realiza la fusión de sensores para proporcionar datos de orientación más precisos.
- Se recolectaron sensores IMU y GPS en un entorno de pastoreo con visión nocturna.
Las señales se muestrearon a 10 Hz (10 muestras por segundo).
La validación se realizó mediante videos sincronizados de cámaras PTZ (Pan-Tilt-Zoom) con visión nocturna, cubriendo un área de pastoreo de 80 hectáreas.
El dataset incluye GPS para contextualizar la ubicación.

Tamaño y Formato:

Total de eventos etiquetados: 441 comportamientos, representando más de 7 horas y 34 minutos de grabación.

Tamaño de datos crudos: >3 GB en archivos CSV (almacenados en tarjetas microSD).
Videos asociados: ~255 GB de 150 horas de footage.

Estructura: Archivos CSV organizados jerárquicamente por categorías de comportamiento (e.g., carpetas para "caminata", "pastoreo", "reposo").
Los nombres de archivos codifican metadatos como número de evento, tipo de comportamiento y timestamps (e.g., formato: evento_numero_comportamiento_inicio-fin.csv).


Variables Principales:

Aceleración lineal: En marcos de referencia corporal (body frame) y mundial (world frame), en 3 ejes (X, Y, Z).
Giroscopio: Velocidad angular en 3 ejes.
Magnetómetro: Campo magnético en 3 ejes.
Orientación: Cuaterniones (para BNO055).
GPS: Latitud, longitud, altitud (opcional para contexto espacial).
Etiquetas: 12 comportamientos, incluyendo los principales: caminata (walking), pastoreo (grazing), reposo (resting), y otros como de pie (standing), lamiendo (licking), etc. Las etiquetas se asignaron manualmente mediante inspección visual de videos.


Acceso: Disponible en el repositorio de GitHub: https://github.com/WASP-lab/db-cow-walking. Incluye un script de ejemplo en Python para clasificación con scikit-learn.


##  Tipos de Variables
El dataset contiene tanto variables numéricas como categóricas.

### Variables Numéricas

- Aceleración lineal (en g): Medida en 3 ejes (X, Y, Z) en marcos de referencia corporal y mundial.
- Velocidad angular del giroscopio (en °/s): Medida en 3 ejes (X, Y, Z).
- Campo magnético del magnetómetro (en µT): Medida en 3 ejes (X, Y, Z).
- Cuaterniones de orientación (solo para BNO055): 4 componentes (w, x, y, z).
- GPS (opcional): Latitud, longitud, altitud.
- Timestamps: Marca temporal de cada registro (en segundos o milisegundos).
- Frecuencia de muestreo: Constante a 10 Hz (10 muestras por segundo.

### Variables Categóricas
- Comportamiento: 12 categorías, incluyendo:
  - Caminata (Walking)
  - Pastoreo (Grazing)
  - Reposo (Resting)
  - De pie (Standing)
  - Lamiendo (Licking)

- Otros (Miscellaneous)
  - Bebiendo (Drinking)
  - Comiendo (Eating)
  - Interacción social (Social Interaction)
  - Moviéndose (Moving)
  - Rascándose (Scratching)
  - Acariciando (Petting)
  - Otros comportamientos no especificados
- Tipo de IMU: Dos tipos (MPU9250 y BNO055).
- ID de Vaca: Identificador único para cada vaca (1 a 10).
- ID de Evento: Identificador único para cada evento registrado.
- Número de Evento: Identificador único para cada evento registrado.
- Duración del Evento: Tiempo total del evento (en hh:mm:ss).
- Fuente del Sensor: Tipo de IMU utilizada (MPU9250 o BNO055).

###  Descripción de las Variables

| Nombre de la variable | Descripción | Escala/Tipo de dato | Unidades de medida |
| :--- | :--- | :--- | :--- |
| **Time** | Timestamp o índice de tiempo que representa el momento en que se registró la muestra de datos. | Numérica Continua | timestamp |
| **BNO055_ARX** | Tasa de rotación absoluta alrededor del eje X del sensor BNO055 (en grados por segundo o radianes por segundo), derivada de la fusión de datos del acelerómetro, giroscopio y magnetómetro. | Numérica Continua | °/s o rad/s |
| **BNO055_ARY** | Tasa de rotación absoluta alrededor del eje Y del sensor BNO055 (en grados por segundo o radianes por segundo). | Numérica Continua | °/s o rad/s |
| **BNO055_ARZ** | Tasa de rotación absoluta alrededor del eje Z del sensor BNO055 (en grados por segundo o radianes por segundo). | Numérica Continua | °/s o rad/s |
| **BNO055_AX** | Aceleración lineal a lo largo del eje X del acelerómetro BNO055 (en m/s²). | Numérica Continua | m/s² |
| **BNO055_AY** | Aceleración lineal a lo largo del eje Y del acelerómetro BNO055 (en m/s²). | Numérica Continua | m/s² |
| **BNO055_AZ** | Aceleración lineal a lo largo del eje Z del acelerómetro BNO055 (en m/s²). | Numérica Continua | m/s² |
| **BNO055_GX** | Velocidad angular (lectura del giroscopio) alrededor del eje X del sensor BNO055 (en grados por segundo). | Numérica Continua | °/s |
| **BNO055_GY** | Velocidad angular alrededor del eje Y del sensor BNO055 (en grados por segundo). | Numérica Continua | °/s |
| **BNO055_GZ** | Velocidad angular alrededor del eje Z del sensor BNO055 (en grados por segundo). | Numérica Continua | °/s |
| **BNO055_MX** | Intensidad del campo magnético a lo largo del eje X del magnetómetro BNO055 (en microteslas o gauss). | Numérica Continua | µT o gauss |
| **BNO055_MY** | Intensidad del campo magnético a lo largo del eje Y del magnetómetro BNO055 (en microteslas o gauss). | Numérica Continua | µT o gauss |
| **BNO055_MZ** | Intensidad del campo magnético a lo largo del eje Z del magnetómetro BNO055 (en microteslas o gauss). | Numérica Continua | µT o gauss |
| **BNO055_Q0** | Componente escalar (w) de la orientación del cuaternión del sensor BNO055, que representa la rotación en el espacio 3D. | Numérica Continua | sin unidades |
| **BNO055_Q1** | Componente X (i) de la orientación del cuaternión del sensor BNO055. | Numérica Continua | sin unidades |
| **BNO055_Q2** | Componente Y (j) de la orientación del cuaternión del sensor BNO055. | Numérica Continua | sin unidades |
| **BNO055_Q3** | Componente Z (k) de la orientación del cuaternión del sensor BNO055. | Numérica Continua | sin unidades |
| **MPU9250_AX** | Aceleración lineal a lo largo del eje X del acelerómetro MPU9250 (en m/s²). | Numérica Continua | m/s² |
| **MPU9250_AY** | Aceleración lineal a lo largo del eje Y del acelerómetro MPU9250 (en m/s²). | Numérica Continua | m/s² |
| **MPU9250_AZ** | Aceleración lineal a lo largo del eje Z del acelerómetro MPU9250 (en m/s²). | Numérica Continua | m/s² |
| **MPU9250_GX** | Velocidad angular alrededor del eje X del giroscopio MPU9250 (en grados por segundo). | Numérica Continua | °/s |
| **MPU9250_GY** | Velocidad angular alrededor del eje Y del giroscopio MPU9250 (en grados por segundo). | Numérica Continua | °/s |
| **MPU9250_GZ** | Velocidad angular alrededor del eje Z del giroscopio MPU9250 (en grados por segundo). | Numérica Continua | °/s |
| **MPU9250_MX** | Intensidad del campo magnético a lo largo del eje X del magnetómetro MPU9250 (en microteslas o gauss). | Numérica Continua | µT o gauss |
| **MPU9250_MY** | Intensidad del campo magnético a lo largo del eje Y del magnetómetro MPU9250 (en microteslas o gauss). | Numérica Continua | µT o gauss |
| **MPU9250_MZ** | Intensidad del campo magnético a lo largo del eje Z del magnetómetro MPU9250 (en microteslas o gauss). | Numérica Continua | µT o gauss |
| **label** | Etiqueta categórica que indica el tipo de actividad, evento o clase para la muestra de datos correspondiente (p. ej., caminando, en reposo, etc., en un contexto de aprendizaje supervisado). | Categórica | categórica |
| **BNO055_acc_magnitude** | Magnitud (norma euclidiana) del vector de aceleración del sensor BNO055, calculada como √(AX² + AY² + AZ²) en m/s². | Numérica Continua | m/s² |
| **BNO055_gyro_magnitude** | Magnitud del vector del giroscopio del sensor BNO055, calculada como √(GX² + GY² + GZ²) en grados por segundo. | Numérica Continua | °/s |
| **MPU9250_acc_magnitude** | Magnitud del vector de aceleración del sensor MPU9250, calculada como √(AX² + AY² + AZ²) en m/s². | Numérica Continua | m/s² |
| **MPU9250_gyro_magnitude** | Magnitud del vector del giroscopio del sensor MPU9250, calculada como √(GX² + GY² + GZ²) en grados por segundo. | Numérica Continua | °/s |
| **acc_mean_window5** | Valor medio de la magnitud de la aceleración en una ventana deslizante de 5 muestras. | Numérica Continua | m/s² |
| **acc_std_window5** | Desviación estándar de la magnitud de la aceleración en una ventana deslizante de 5 muestras. | Numérica Continua | m/s² |
| **gyro_mean_window5** | Valor medio de la magnitud del giroscopio en una ventana deslizante de 5 muestras. | Numérica Continua | °/s |
| **gyro_std_window5** | Desviación estándar de la magnitud del giroscopio en una ventana deslizante de 5 muestras. | Numérica Continua | °/s |
| **inquietud_alta** | Indicador binario o categórico para alta inquietud o agitación, derivado de umbrales en los datos del sensor (p. ej., alta variabilidad en las lecturas de aceleración o giroscopio). | Categórica | binaria |
| **actividad_extrema** | Indicador binario o categórico para niveles de actividad extremos, basado en la superación de umbrales predefinidos en las magnitudes del sensor de movimiento. | Categórica | binaria |
"""

st.markdown(multi)

# Estadísticas descriptivas numéricas
col_numericas = df.select_dtypes(
    include=[np.number]).columns.drop('label', errors='ignore')
st.write("Estadísticas numéricas:\n", df[col_numericas].describe())

# Visualización distribución labels
plt.figure()
sns.countplot(data=df, x='label')
plt.title('Distribución de Labels')
plt.xticks(rotation=45)
st.pyplot(plt)

# Heatmap correlaciones (selección de cols para evitar too many)
seleccion_numerica = col_numericas[:10]  # Limitar para visualización
corr = df[seleccion_numerica].corr()
plt.figure()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlaciones (primeras 10 cols numéricas)')
st.pyplot(plt)
