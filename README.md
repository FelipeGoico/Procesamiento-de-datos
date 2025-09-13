# Análisis Exploratorio y Reducción de Dimensionalidad

# db-cow-walking-IoT Base Comportamientos en Vacas de Pastoreo

# 

# 📖 Descripción del Proyecto

# Este repositorio contiene una aplicación interactiva desarrollada en Streamlit para el análisis exploratorio de datos (EDA) y reducción de dimensionalidad aplicada a un dataset público de comportamientos en vacas de pastoreo (db-cow-walking-IoT). El dataset original, basado en sensores IMU (MPU9250 y BNO055) y validación por video PTZ, registra actividades como caminar, pastorear y descansar en 10 vacas lecheras.

# Objetivos Principales

# 

# Generar variables derivadas: Magnitudes de aceleración/giroscopio, estadísticas en ventanas deslizantes e indicadores de anomalías (e.g., inquietud alta para detectar espasmos por moscas).

# Aplicar técnicas de reducción de dimensionalidad: PCA (no supervisado), LDA (supervisado), t-SNE y UMAP para visualización en 2D/3D.

# Interfaz interactiva: Exploración visual de patrones de comportamiento, con potencial extensión a monitoreo con drones.

# Enfoque educativo: Proyecto final de Magíster en Data Science, integrando preprocesamiento, modelado y visualización.

# 

# El proyecto reutiliza el dataset original de un estudio en la Granja Experimental Maquehue (Universidad de La Frontera, Temuco, Chile), extendiéndolo hacia detección temprana de anomalías sanitarias en ganadería.

# Dataset

# 

# Fuente: Artículo original en Frontiers in Veterinary Science (acceso abierto).

# Tamaño: >3 GB crudos, 441 eventos etiquetados, >7 horas de grabación a 10 Hz.

# Variables clave: Aceleración (AX, AY, AZ), giroscopio (GX, GY, GZ), magnetómetro, cuaterniones y GPS en marcos body/world.

# Archivo procesado: data\_processed.csv (incluido; generado a partir de CSVs originales por comportamiento).

