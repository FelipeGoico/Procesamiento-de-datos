from app import st, df
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# Configuración página
# ===========================
st.set_page_config(
    page_title="EDA - Dataset db-cow-walking-IoT",
    page_icon=":mag:",
    layout="wide"
)

# ===========================
# Banner
# ===========================
st.markdown("""
<div style="background-color:#1f4e79; padding: 18px; border-radius: 10px; text-align:center; color: white;">
    <h1 style="margin:0;">🔍 Análisis Exploratorio de Datos (EDA)</h1>
    <p style="margin:0; font-size:18px;">Dataset <i>db-cow-walking-IoT</i></p>
</div>
""", unsafe_allow_html=True)

# ===========================
# Introducción con continuidad
# ===========================
st.markdown("""
<div style="text-align: justify; margin-top: 20px;">
El presente Análisis Exploratorio de Datos (EDA) se basa en el dataset original 
<b>db-cow-walking-IoT</b>, que recoge información de sensores inerciales (IMU) y GPS en vacas lecheras de pastoreo.  
Este conjunto de datos ha permitido identificar y clasificar comportamientos como caminar, pastorear o descansar.  

Dando continuidad al proyecto anterior, utilizaremos este EDA como <b>punto de partida</b> para profundizar en nuevas líneas de investigación.  
Nuestro objetivo es ampliar la detección hacia <b>comportamientos anómalos</b>, como espasmos o sacudidas vinculadas a la presencia de moscas, en especial en zonas del lomo y los cuernos.  
La incorporación de estas nuevas variables derivadas no solo enriquecerá el dataset, sino que también abrirá la posibilidad de <b>automatizar la detección temprana</b> de posibles problemas de salud animal, mejorando así la gestión en la ganadería moderna.  
</div>
""", unsafe_allow_html=True)

# ===========================
# Dataset y variables
# ===========================
st.markdown("""
## 📊 Tipos de Variables en el Dataset
El dataset contiene tanto variables numéricas como categóricas que permiten representar el comportamiento de las vacas.
""")

# Variables numéricas y categóricas
multi = """
### Variables Numéricas
- Aceleración lineal (en g): Medida en 3 ejes (X, Y, Z) en marcos de referencia corporal y mundial.
- Velocidad angular del giroscopio (en °/s): Medida en 3 ejes (X, Y, Z).
- Campo magnético del magnetómetro (en µT): Medida en 3 ejes (X, Y, Z).
- Cuaterniones de orientación (solo para BNO055): 4 componentes (w, x, y, z).
- GPS (opcional): Latitud, longitud, altitud.
- Timestamps: Marca temporal de cada registro (en segundos o milisegundos).
- Frecuencia de muestreo: Constante a 10 Hz (10 muestras por segundo).

### Variables Categóricas
- **Comportamiento**: 12 categorías principales:
  - Caminata (Walking)
  - Pastoreo (Grazing)
  - Reposo (Resting)
  - De pie (Standing)
  - Lamiendo (Licking)
  - Bebiendo (Drinking)
  - Comiendo (Eating)
  - Interacción social (Social Interaction)
  - Moviéndose (Moving)
  - Rascándose (Scratching)
  - Acariciando (Petting)
  - Otros comportamientos no especificados (Miscellaneous)
- **Tipo de IMU**: Dos tipos (MPU9250 y BNO055).
- **ID de Vaca**: Identificador único para cada vaca (1 a 10).
- **ID de Evento**: Identificador único para cada evento registrado.
- **Número de Evento**: Contador secuencial de eventos por vaca.
- **Duración del Evento**: Tiempo total del evento (en hh:mm:ss).
- **Fuente del Sensor**: Tipo de IMU utilizada (MPU9250 o BNO055).
"""
st.markdown(multi)


# ===========================
# Estadísticas descriptivas
# ===========================
col_numericas = df.select_dtypes(include=[np.number]).columns.drop('label', errors='ignore')
st.markdown("### 📈 Estadísticas Descriptivas de Variables Numéricas")
st.write(df[col_numericas].describe())

# ===========================
# Visualización: Distribución de labels
# ===========================
st.markdown("### 🐄 Distribución de Comportamientos (Labels)")
plt.figure()
sns.countplot(data=df, x='label')
plt.title('Distribución de Labels')
plt.xticks(rotation=45)
st.pyplot(plt)

# ===========================
# Visualización: Correlaciones
# ===========================
st.markdown("### 🔗 Correlaciones entre Variables Numéricas")
seleccion_numerica = col_numericas[:10]  # limitar para no saturar
corr = df[seleccion_numerica].corr()
plt.figure()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlaciones (primeras 10 cols numéricas)')
st.pyplot(plt)
