db-cow-walking-IoT: Análisis Exploratorio y Reducción de Dimensionalidad de Comportamientos en Vacas de Pastoreo

📖 Descripción del Proyecto
Este repositorio contiene una aplicación interactiva desarrollada en Streamlit para el análisis exploratorio de datos (EDA) y reducción de dimensionalidad aplicada a un dataset público de comportamientos en vacas de pastoreo (db-cow-walking-IoT). El dataset original, basado en sensores IMU (MPU9250 y BNO055) y validación por video PTZ, registra actividades como caminar, pastorear y descansar en 10 vacas lecheras.
Objetivos Principales

Generar variables derivadas: Magnitudes de aceleración/giroscopio, estadísticas en ventanas deslizantes e indicadores de anomalías (e.g., inquietud alta para detectar espasmos por moscas).
Aplicar técnicas de reducción de dimensionalidad: PCA (no supervisado), LDA (supervisado), t-SNE y UMAP para visualización en 2D/3D.
Interfaz interactiva: Exploración visual de patrones de comportamiento, con potencial extensión a monitoreo con drones.
Enfoque educativo: Proyecto final de Magíster en Data Science, integrando preprocesamiento, modelado y visualización.

El proyecto reutiliza el dataset original de un estudio en la Granja Experimental Maquehue (Universidad de La Frontera, Temuco, Chile), extendiéndolo hacia detección temprana de anomalías sanitarias en ganadería.
Dataset

## Descripción del Dataset

- **Fuente**: Estudio publicado en *Frontiers in Veterinary Science* (2023): "A dataset for detecting walking, grazing, and resting behaviors in free-grazing cattle using IoT collar IMU signals".
- **Contenido**: 
  - 441 eventos etiquetados (caminar, pastorear, reposar).
  - Más de 7 horas de grabación continua.
  - >3 GB de datos crudos.
- **Variables Principales**:
  - Aceleración lineal (3 ejes, body/world frame).
  - Velocidad angular (giroscopio, 3 ejes).
  - Magnetómetro (3 ejes).
  - Orientación absoluta (cuaterniones del BNO055).
- **Variables Derivadas** (generadas en este proyecto):
  - Magnitudes euclidianas (aceleración y giroscopio).
  - Estadísticas en ventana deslizante (media y desviación estándar).
  - Indicadores categóricos (inquietud alta, actividad extrema).
- **Formato**: Archivos CSV procesados en `data_processed.csv` (disponible para descarga en la app).
- **Ubicación de Recolección**: Granja Experimental Maquehue, Universidad de La Frontera, Temuco, Chile.
- **Sujetos**: 10 vacas lecheras con collares IoT en el cuello.

El dataset permite explorar patrones de comportamiento y detectar anomalías para aplicaciones en monitoreo ganadero automatizado.

## Requisitos


- Bibliotecas clave: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `umap-learn`, `scikit-umap`.
- Opcional: Entorno virtual  `conda`

## Instrucciones de Ejecución

1. **Clona o descarga el repositorio**:
   - Asegúrate de tener la estructura de carpetas: `app.py` en la raíz, `pages/` con subpáginas (1_EDA.py, etc.), `data/` con `data_processed.csv`, e `img/` con imágenes.

2. **Navega al directorio del proyecto**:
   ```bash
   cd path/proyecto
   ```

3. **Ejecuta la aplicación Streamlit**:
   ```bash
   streamlit run app.py
   ```

4. **Accede a la interfaz**:
   - Abre tu navegador en `http://localhost:8501`.
   - Explora secciones: Introducción, Proyecto Original, Variables Derivadas, Metodología, EDA.
   - Usa botones para navegar a páginas específicas (EDA, LDA, PCA, t-SNE, UMAP).
   - Descarga el dataset procesado desde el botón "📥 Descargar Datos CSV".

5. **Personalización**:
   - Edita `data_loader.py` para cargar datos personalizados.
   - Ajusta parámetros en las páginas (e.g., perplexity en t-SNE) vía controles interactivos.

## Estructura del Proyecto

```
proyecto/
├── app.py                  # Página principal (introducción y navegación)
├── pages/                  # Subpáginas de análisis
│   ├── 1_EDA.py           # Análisis Exploratorio
│   ├── 2_LDA.py           # Análisis Discriminante Lineal
│   ├── 3_PCA.py           # Análisis de Componentes Principales
│   ├── 4_t_SNE.py         # t-Distributed Stochastic Neighbor Embedding
│   └── 5_UMAP.py          # Uniform Manifold Approximation and Projection
├── data/                   # Datos
│   └── data_processed.csv  # Dataset procesado
├── img/                    # Imágenes
│   ├── fvets-12-1630083-g002.jpg
│   ├── fvets-12-1630083-g003.jpg
│   ├── fvets-12-1630083-g004.jpg
│   └── fvets-12-1630083-g005.jpg
├── data_loader.py          # Funciones de carga y preparación de datos
└── README.md               # Este archivo
```

¡Disfruta explorando los datos de comportamiento animal! 🐄📊