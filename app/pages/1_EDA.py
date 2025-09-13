import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA


df = st.session_state.df


# ===========================
# Estilo CSS personalizado
# ===========================
st.markdown("""
<style>
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 15px;
    font-family: 'Segoe UI', sans-serif;
}
th {
    background-color: #1f4e79;
    color: white;
    text-align: center;
    padding: 8px;
    border: 1px solid #ddd;
}
td {
    background-color: #f9f9f9;
    text-align: left;
    padding: 8px;
    border: 1px solid #ddd;
}
tr:nth-child(even) td {
    background-color: #f1f1f1;
}
</style>
""", unsafe_allow_html=True)


# sube desde pages -> app -> Procesamiento-de-datos
BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMG_DIR = BASE_DIR / "img"
IMAGE_PATH_00 = IMG_DIR / "fvets-12-1630083-g000.jpg"


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


# ==========================================================
# ==================== Preparación de Datos ================
# ==========================================================
st.markdown("""
## ⚙️ Tratamiento y Preparación de los Datos

<div style="text-align: justify;">
Los datos originales se encontraban organizados en múltiples carpetas, donde cada una representaba un comportamiento distinto 
(caminata, pastoreo, reposo, entre otros). Cada carpeta contenía varios archivos <code>.csv</code> con registros capturados 
desde los sensores <b>MPU9250</b> y <b>BNO055</b>.  

Para unificar y estructurar esta información, se diseñó un <b>método adaptable</b> que permite:  

1. Recorrer automáticamente todas las carpetas y archivos.  
2. Incorporar una columna <code>label</code> con el nombre de la carpeta (comportamiento observado).  
3. Calcular magnitudes derivadas de aceleración y giroscopio.  
4. Generar estadísticas de ventana temporal (rolling window) para capturar variaciones dinámicas.  
5. Crear indicadores de alerta como <i>inquietud alta</i> o <i>actividad extrema</i>.  

Este enfoque permitió transformar un conjunto de archivos dispersos en un dataset unificado, consistente y listo para el análisis exploratorio.  
</div>
""", unsafe_allow_html=True)


# ===========================
# Mostrar código usado (colapsable)
# ===========================
with st.expander("📜 Ver código usado en el tratamiento de los datos"):
    st.code("""
import os
import pandas as pd 
import numpy as np

def load_data_from_folders(base_path):
    df_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                df = pd.read_csv(file_path)
                df['label'] = folder_name
                df_list.append(df)
    df_full = pd.concat(df_list, ignore_index=True)
    return df_full

df_full = load_data_from_folders('C:/ruta/base/db-cow-walking-IoT')
df = df_full.copy()

# Magnitudes de sensores
df['BNO055_acc_magnitude'] = np.sqrt(df['BNO055_AX']**2 + df['BNO055_AY']**2 + df['BNO055_AZ']**2)
df['BNO055_gyro_magnitude'] = np.sqrt(df['BNO055_GX']**2 + df['BNO055_GY']**2 + df['BNO055_GZ']**2)
df['MPU9250_acc_magnitude'] = np.sqrt(df['MPU9250_AX']**2 + df['MPU9250_AY']**2 + df['MPU9250_AZ']**2)
df['MPU9250_gyro_magnitude'] = np.sqrt(df['MPU9250_GX']**2 + df['MPU9250_GY']**2 + df['MPU9250_GZ']**2)

# Estadísticas por ventana
window_size = 5
df['acc_mean_window5'] = df['BNO055_acc_magnitude'].rolling(window_size).mean()
df['acc_std_window5'] = df['BNO055_acc_magnitude'].rolling(window_size).std()
df['gyro_mean_window5'] = df['BNO055_gyro_magnitude'].rolling(window_size).mean()
df['gyro_std_window5'] = df['BNO055_gyro_magnitude'].rolling(window_size).std()

# Indicadores de alerta
df['inquietud_alta'] = (
    (df['label'] == 'Resting') & 
    (df['BNO055_gyro_magnitude'] > df['BNO055_gyro_magnitude'].quantile(0.95))
).astype(int)

df['actividad_extrema'] = (
    df['BNO055_acc_magnitude'] > df['BNO055_acc_magnitude'].quantile(0.95)
).astype(int)
    """, language="python")


st.image(str(IMAGE_PATH_00),
         caption="Estructura de carpetas y archivos del dataset original", use_container_width=True)

# ===========================
# Dataset y variables
# ===========================
st.markdown("""
## 📊 Tipos de Variables en el Dataset
El dataset contiene tanto variables numéricas como categóricas que permiten representar el comportamiento de las vacas.
""")


# ===========================
# Sección descriptiva
# ===========================
st.markdown("## 📑 Descripción de las Variables")
st.markdown("""
Esta sección presenta una descripción detallada de las variables contenidas en el dataset
**db-cow-walking-IoT**, clasificadas según su origen y naturaleza.
Se busca mantener una presentación clara, concisa y profesional para apoyar la investigación.
""")


# ===========================
# Variables de Tiempo
# ===========================
with st.expander("🕒 Variables de Tiempo", expanded=True):
    st.markdown("""
| Nombre | Descripción | Tipo de dato | Unidad |
| :--- | :--- | :--- | :--- |
| **Time** | Timestamp que indica el momento de registro de la muestra. | Numérica Continua | timestamp |
""")

# ===========================
# Variables del Sensor BNO055
# ===========================
with st.expander("📡 Variables del Sensor BNO055"):
    st.markdown("""
| Nombre | Descripción | Tipo de dato | Unidad |
| :--- | :--- | :--- | :--- |
| **BNO055_ARX** | Tasa de rotación absoluta eje X. | Numérica Continua | °/s o rad/s |
| **BNO055_ARY** | Tasa de rotación absoluta eje Y. | Numérica Continua | °/s o rad/s |
| **BNO055_ARZ** | Tasa de rotación absoluta eje Z. | Numérica Continua | °/s o rad/s |
| **BNO055_AX** | Aceleración lineal eje X. | Numérica Continua | m/s² |
| **BNO055_AY** | Aceleración lineal eje Y. | Numérica Continua | m/s² |
| **BNO055_AZ** | Aceleración lineal eje Z. | Numérica Continua | m/s² |
| **BNO055_GX** | Velocidad angular eje X. | Numérica Continua | °/s |
| **BNO055_GY** | Velocidad angular eje Y. | Numérica Continua | °/s |
| **BNO055_GZ** | Velocidad angular eje Z. | Numérica Continua | °/s |
| **BNO055_MX** | Campo magnético eje X. | Numérica Continua | µT |
| **BNO055_MY** | Campo magnético eje Y. | Numérica Continua | µT |
| **BNO055_MZ** | Campo magnético eje Z. | Numérica Continua | µT |
| **BNO055_Q0** | Componente escalar (w) del cuaternión. | Numérica Continua | — |
| **BNO055_Q1** | Componente X (i) del cuaternión. | Numérica Continua | — |
| **BNO055_Q2** | Componente Y (j) del cuaternión. | Numérica Continua | — |
| **BNO055_Q3** | Componente Z (k) del cuaternión. | Numérica Continua | — |
""")

# ===========================
# Variables del Sensor MPU9250
# ===========================
with st.expander("⚙️ Variables del Sensor MPU9250"):
    st.markdown("""
| Nombre | Descripción | Tipo de dato | Unidad |
| :--- | :--- | :--- | :--- |
| **MPU9250_AX** | Aceleración lineal eje X. | Numérica Continua | m/s² |
| **MPU9250_AY** | Aceleración lineal eje Y. | Numérica Continua | m/s² |
| **MPU9250_AZ** | Aceleración lineal eje Z. | Numérica Continua | m/s² |
| **MPU9250_GX** | Velocidad angular eje X. | Numérica Continua | °/s |
| **MPU9250_GY** | Velocidad angular eje Y. | Numérica Continua | °/s |
| **MPU9250_GZ** | Velocidad angular eje Z. | Numérica Continua | °/s |
| **MPU9250_MX** | Campo magnético eje X. | Numérica Continua | µT |
| **MPU9250_MY** | Campo magnético eje Y. | Numérica Continua | µT |
| **MPU9250_MZ** | Campo magnético eje Z. | Numérica Continua | µT |
""")

# ===========================
# Variables Derivadas
# ===========================
with st.expander("🧮 Variables Derivadas"):
    st.markdown("""
| Nombre | Descripción | Tipo de dato | Unidad |
| :--- | :--- | :--- | :--- |
| **BNO055_acc_magnitude** | Magnitud de la aceleración BNO055. | Numérica Continua | m/s² |
| **BNO055_gyro_magnitude** | Magnitud del giroscopio BNO055. | Numérica Continua | °/s |
| **MPU9250_acc_magnitude** | Magnitud de la aceleración MPU9250. | Numérica Continua | m/s² |
| **MPU9250_gyro_magnitude** | Magnitud del giroscopio MPU9250. | Numérica Continua | °/s |
| **acc_mean_window5** | Media de aceleración en ventana de 5. | Numérica Continua | m/s² |
| **acc_std_window5** | Desviación estándar de aceleración en ventana de 5. | Numérica Continua | m/s² |
| **gyro_mean_window5** | Media del giroscopio en ventana de 5. | Numérica Continua | °/s |
| **gyro_std_window5** | Desviación estándar del giroscopio en ventana de 5. | Numérica Continua | °/s |
""")

# ===========================
# Variables de Etiqueta
# ===========================
with st.expander("🏷️ Variables de Etiqueta"):
    st.markdown("""
| Nombre | Descripción | Tipo de dato | Unidad |
| :--- | :--- | :--- | :--- |
| **label** | Clase categórica del comportamiento observado. | Categórica | — |
| **inquietud_alta** | Indicador binario de alta inquietud. | Categórica | 0 / 1 |
| **actividad_extrema** | Indicador binario de actividad extrema. | Categórica | 0 / 1 |
""")

# ===========================
# Validación de NA / Null
# ===========================
st.markdown("## ⚠️ Validación de valores faltantes (NA / Null)")

# Cantidad de NA por columna
na_count = df.isna().sum()
# Porcentaje de NA por columna
na_percent = (na_count / len(df)) * 100

na_df = pd.DataFrame({
    "Columna": df.columns,
    "NA_count": na_count,
    "NA_percent": na_percent.round(2)
}).sort_values(by="NA_percent", ascending=False)

st.dataframe(na_df, use_container_width=True)

# Mensaje de alerta si hay NA
threshold = 5  # porcentaje considerado crítico
if any(na_percent > threshold):
    st.warning(f"Hay columnas con más del {threshold}% de valores faltantes. Considera imputar o limpiar antes del análisis.")
else:
    st.success("No se detectan valores faltantes relevantes.")

 
st.markdown("""
### 📊 Distribución de Variables Numéricas

Antes de explorar las relaciones entre variables, es importante conocer la distribución individual de cada columna numérica. 
Esto permite detectar sesgos, valores atípicos o rangos inesperados, y sirve como base para análisis posteriores como correlaciones o grafo de variables.
""")

fig, col_numericas = st.session_state.graphs
st.pyplot(fig)

st.markdown("### 📊 Mapa de Correlación de Variables Numéricas por columna")

# Seleccionar variables numéricas
col_numericas = df.select_dtypes(
    include=[np.number]).columns.drop('label', errors='ignore')

# Calcular matriz de correlación
corr_matrix = df[col_numericas].corr()

# Triangular inferior: ponemos NaN en la parte superior
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_masked = corr_matrix.mask(mask)

# Crear heatmap interactivo con Plotly
fig = px.imshow(
    corr_masked,
    text_auto='.2f',       # mostrar valores en cada celda
    aspect="auto",
    color_continuous_scale='RdBu_r',
    origin='upper'
)

# Reducir tamaño de texto dentro de los cuadros
# ajusta según convenga, 10 es ~40% más pequeño que el default
fig.update_traces(textfont_size=10)

# Mejorar layout
fig.update_layout(
    title='Mapa de Correlación de Variables Numéricas (Triangular Inferior)',
    xaxis_title='Variables',
    yaxis_title='Variables',
    xaxis_tickangle=-45,
    width=900,
    height=900
)

st.plotly_chart(fig, use_container_width=True)



# ==========================================================
# Título y descripción
# ==========================================================
st.markdown("### 🕸️ Grafo de Correlación de Variables")
st.markdown("""
<div style="text-align: justify; margin-top: 10px;">
Esta visualización presenta las correlaciones más fuertes como una red, donde cada nodo es una variable. 
Una línea (arista) entre dos nodos indica una correlación significativa (positiva o negativa). Esto reduce la 
saturación visual y ayuda a identificar rápidamente las relaciones clave.
</div>
""", unsafe_allow_html=True)

# Leyenda clara separada
st.markdown("""
**Leyenda:**  
<b style="color:red;">Rojo:</b> Correlación positiva fuerte  
<b style="color:blue;">Azul:</b> Correlación negativa fuerte  
Solo se muestran correlaciones > 0.7 o < -0.7
""", unsafe_allow_html=True)

# ==========================================================
# Preparar matriz de correlación y edges
# ==========================================================
col_numericas = df.select_dtypes(include=[np.number]).columns.drop('label', errors='ignore')
corr_matrix = df[col_numericas].corr()

threshold = 0.7
edges = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            edges.append((var1, var2, {'correlation': corr_value}))

# ==========================================================
# Crear grafo y posiciones
# ==========================================================
G = nx.Graph()
G.add_nodes_from(corr_matrix.columns)
G.add_edges_from(edges)

pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)  # mayor separación

# ==========================================================
# Preparar trazos para aristas
# ==========================================================
pos_edges_x, pos_edges_y, neg_edges_x, neg_edges_y = [], [], [], []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    corr = G.get_edge_data(edge[0], edge[1])['correlation']
    if corr > 0:
        pos_edges_x.extend([x0, x1, None])
        pos_edges_y.extend([y0, y1, None])
    else:
        neg_edges_x.extend([x0, x1, None])
        neg_edges_y.extend([y0, y1, None])

pos_edge_trace = go.Scatter(
    x=pos_edges_x, y=pos_edges_y,
    line=dict(width=1.5, color='rgba(255,0,0,0.7)'),
    mode='lines',
    hoverinfo='none',
    name='Correlación Positiva'
)
neg_edge_trace = go.Scatter(
    x=neg_edges_x, y=neg_edges_y,
    line=dict(width=1.5, color='rgba(0,0,255,0.7)'),
    mode='lines',
    hoverinfo='none',
    name='Correlación Negativa'
)

# ==========================================================
# Preparar nodos (texto flotante)
# ==========================================================
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_text = []
for node, neighbors in G.adjacency():
    correlations = []
    for neighbor in neighbors:
        corr_val = G.get_edge_data(node, neighbor)['correlation']
        correlations.append(f"{neighbor}: {corr_val:.2f}")
    hover_text = f"<b>{node}</b><br>Correlaciones fuertes:<br>" + "<br>".join(correlations)
    node_text.append(hover_text)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[node for node in G.nodes()],     # solo el nombre visible sobre el nodo
    hovertext=node_text,                   # información detallada al pasar el cursor
    hoverinfo='text',
    textposition='top center',
    marker=dict(
        size=10,
        color='DarkBlue',
        line=dict(color='black', width=1)
    ),
    textfont=dict(size=10, color='black'),
    name='Variables'
)

# ==========================================================
# Layout final con marco
# ==========================================================
fig = go.Figure(
    data=[pos_edge_trace, neg_edge_trace, node_trace],
    layout=go.Layout(
        title=dict(text='', font=dict(size=16)),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0, y0=0,
                x1=1, y1=1,
                line=dict(color="black", width=2),
                fillcolor="white",
                layer="below"
            )
        ]
    )
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# ==================== Preparación de Datos ====================
# ==========================================================
st.markdown("""
## 🛠️ Preparación de Datos para Modelamiento

Después de explorar y analizar el dataset, es necesario preparar los datos antes de aplicar modelos de machine learning.  
Esto incluye:

1. Imputación de valores faltantes.  
2. Escalado de las variables numéricas.  
3. Codificación de la variable target y partición Train/Test.
""")

# --------------------------
# 4.1 Imputación (KNNImputer)
# --------------------------
st.markdown("### 4.1 Imputación de Valores Faltantes (KNNImputer)")

# Seleccionar features numéricas (excluir Time y label si no son numéricas)
feature_cols = [col for col in df.columns if col != 'label' and df[col].dtype in [np.float64, np.int64]]
X = df[feature_cols]
y = df['label']

# Imputación
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

st.write("✅ Imputación completada. Se utilizó KNNImputer considerando correlación entre sensores.")
st.write("% NA después de imputación:", round(X_imputed.isna().sum().sum()/X_imputed.size*100, 2), "%")

# --------------------------
# 4.2 Escalado (StandardScaler)
# --------------------------
st.markdown("### 4.2 Escalado de Variables Numéricas (StandardScaler)")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols)

st.write("✅ Escalado completado. Las variables numéricas ahora tienen media=0 y desviación estándar=1.")

# --------------------------
# 4.3 Codificación y Train/Test Split
# --------------------------
st.markdown("### 4.3 Codificación de Etiqueta y División Train/Test")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

st.write(f"✅ División completada: Train={X_train.shape}, Test={X_test.shape}")
st.write("La variable target ha sido codificada con LabelEncoder para su uso en modelos categóricos.")

