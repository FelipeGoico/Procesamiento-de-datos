<<<<<<< HEAD
import app import st
=======
from app import *
>>>>>>> origin/main


st.write("""# Resumen del Examen Final
Este examen final abarca diversas técnicas de procesamiento de datos y análisis estadístico, incluyendo Análisis
Exploratorio de Datos (EDA), Análisis Discriminante Lineal (LDA) y Análisis de Componentes Principales (PCA). A continuación,
se presenta un resumen de cada una de estas técnicas y sus aplicaciones.
""")
st.set_page_config(page_title="Resumen",
                   page_icon=":bar_chart:", layout="wide")
<<<<<<< HEAD
=======
st.html("""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Análisis Exploratorio y Proyección: Proyecto IoT Collar – db-cow-walking-IoT</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #2c3e50;
      line-height: 1.6;
      margin: 40px;
    }
    h1, h2, h3 {
      color: #1e3799;
    }
    .section {
      margin-bottom: 40px;
    }
    .image-center {
      text-align: center;
      margin: 20px 0;
    }
    .image-center img {
      max-width: 100%;
      height: auto;
      border: 2px solid #1e3799;
      border-radius: 8px;
    }
    .caption {
      font-size: 0.9em;
      color: #34495e;
    }
    .formula-block {
      background: #f6f6f6;
      padding: 15px;
      border-left: 5px solid #1e3799;
      margin: 15px 0;
      font-style: italic;
    }
  </style>
</head>
<body>

  <h1>Análisis Exploratorio & Proyección: db-cow-walking-IoT</h1>

  <div class="section" id="introduccion">
    <h2>1. Introducción</h2>
    <p>
      Este proyecto se basa en el estudio de un dataset público reciente que registra comportamientos de vacas de pastoreo — caminata, pastoreo y reposo — mediante collares IoT con sensores IMU
      (MPU9250 y BNO055) y validación por video PTZ. El paper original aporta un dataset estructurado con 441 eventos etiquetados, más de 7 h de grabación, y más de 3 GB de datos crudos, lo que lo convierte en un punto de partida sólido. :contentReference[oaicite:0]{index=0}
    </p>
    <p>
      Nuestro objetivo específico es reutilizar este dataset para: (1) generar nuevas variables derivadas, (2) aplicar reducción de dimensionalidad, y (3) crear una interfaz interactiva en Streamlit que posibilite análisis exploratorio y visualización avanzada, con visión futura hacia integración con drones para monitoreo territorial.
    </p>
  </div>

  <div class="section" id="dataset_original">
    <h2>2. Proyecto Original: Descripción técnica</h2>
    <h3>2.1 Recolección de datos</h3>
    <ul>
      <li>Ubicación: Granja Experimental Maquehue, Universidad de La Frontera, Temuco, Chile. :contentReference[oaicite:1]{index=1}</li>
      <li>Sujetos: 10 vacas lecheras con collares IoT, colocados en el cuello. :contentReference[oaicite:2]{index=2}</li>
      <li>Sensores:  
        <ul>
          <li><b>MPU9250</b>: IMU de 9 ejes, económico, registra aceleración, giroscopio y magnetómetro. :contentReference[oaicite:3]{index=3}</li>
          <li><b>BNO055</b>: IMU fusionado, permite medir orientación como cuaterniones para estimar marco mundial/world frame. :contentReference[oaicite:4]{index=4}</li>
        </ul>
      </li>
      <li>Frecuencia de muestreo: 10 Hz. :contentReference[oaicite:5]{index=5}</li>
      <li>Validación visual: cámaras PTZ con visión nocturna, sincronizadas con las señales de los sensores. :contentReference[oaicite:6]{index=6}</li>
      <li>Estructura de almacenamiento: archivos CSV por evento (nombre codificado: evento, comportamiento, identificador de vaca, fecha–hora), organizados jerárquicamente por tipo de comportamiento. :contentReference[oaicite:7]{index=7}</li>
    </ul>

    <h3>2.2 Variables originales y procesamiento</h3>
    <p>
      Las señales registradas incluyen aceleración lineal, velocidad angular, magnetómetro, cuaterniones (solo BNO055) y GPS. Variables medidas tanto en marco corporal (“body frame”) como mundial (“world frame”) permiten estimaciones más robustas. :contentReference[oaicite:8]{index=8}
    </p>
    <p>
      Para transformar aceleraciones del marco corporal al mundial se usan matrices de rotación derivadas de las orientaciones dadas por los cuaterniones. Además, los datos crudos deben escalarse a unidades físicas mediante factores de escala según sensibilidad y rango del sensor. :contentReference[oaicite:9]{index=9}
    </p>
  </div>

  <div class="section" id="imagenes">
    <h2>3. Ilustraciones relevantes</h2>

    <div class="image-center">
      <img src="img/fvets-12-1630083-g002.jpg" alt="Mapa de localidad Maquehue" >
      <p class="caption">Figura: Localización de la Granja Experimental Maquehue, Temuco, Chile (Mapa usado en publicación original).</p>
    </div>

    <!-- Puedes agregar más imágenes si las tienes (sensor, montaje del collar, etc.) -->
  </div>

  <div class="section" id="variables_derivadas">
    <h2>4. Variables derivadas & Reducción de Dimensionalidad</h2>
    <p>
      Basándonos en los datos originales, hemos generado variables adicionales útiles para análisis y modelado:
    </p>
    <ul>
      <li>Magnitudes de aceleración y giroscopio:

        """)
st.latex(r"""\text{acc\_magnitude} = \sqrt{AX^2 + AY^2 + AZ^2}""")
st.latex(r"""\text{gyro\_magnitude} = \sqrt{GX^2 + GY^2 + GZ^2}""")
st.html(
        """

      </li>
      <li>Estadísticos móviles (ventanas deslizantes, por ejemplo de 5 muestras) para media y desviación estándar de aceleración y giro.</li>
      <li>Indicadores binarios: “actividad extrema”, “inquietud”, etc., usando umbrales basados en la variabilidad de señales.</li>
    </ul>
    <p>
      Para la reducción de dimensionalidad, consideraremos técnicas como <b>PCA (Análisis de Componentes Principales)</b>, <b>Selección de características</b> (por ejemplo, importancia por permutación) y/o técnicas de embedding (t-SNE, UMAP) para visualización de relaciones entre eventos.
    </p>
  </div>

  <div class="section" id="metodologia_propuesta">
    <h2>5. Metodología propuesta para mi proyecto</h2>
    <ol>
      <li>Preprocesamiento: limpiar señales, sincronizar timestamps, escalar valores, detectar y manejar outliers.</li>
      <li>Generación de variables derivadas (como en sección anterior), y creación de dataset estructurado.</li>
      <li>División de datos para entrenamiento / validación, experimentos con diferentes algoritmos supervisados: SVM, Random Forest, etc.</li>
      <li>Reducción de dimensionalidad y visualización en Streamlit, permitiendo explorar comportamientos por vaca, por evento, y visualizar componentes principales.</li>
      <li>Proyección futura hacia drones: usar sensores locales + cámara aérea + GPS para ampliar cobertura, capturar variables ambientales (temperatura, viento, visual), sincronizar con los datos del collar, para detectar patrones más complejos (como posibles signos de estrés o enfermedad).</li>
    </ol>
  </div>

  <div class="section" id="conclusion">
    <h2>6. Conclusiones</h2>
    <p>
      El dataset original proporciona una base sólida: gran riqueza en señales IMU, etiquetado de comportamientos, estructura clara y buenos ejemplos de uso en clasificación. Sin embargo, mi proyecto se diferencia al incorporar variables derivadas, visualizar de forma interactiva, y preparar el camino hacia integración con sistemas aéreos (drones) para monitoreo territorial.
    </p>
    <p>
      Se espera que, al aplicar reducción de dimensionalidad y crear una app en Streamlit, los resultados sean fácilmente interpretables por distintos actores (investigadores, veterinarios, productores), además de escalables. La colaboración futura podría involucrar recopilación adicional de datos, sensores ambientales, y validación mediante drones para mejorar la resolución espacial del monitoreo.
    </p>
  </div>

</body>
</html>
""")
>>>>>>> origin/main
