from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import matplotlib.pyplot as plt
import io

def create_project_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=30)
    
    elementos = []

    # Estilos
    styles = getSampleStyleSheet()
    # Evitar errores si los estilos ya existen
    if 'Heading2Custom' not in styles:
        styles.add(ParagraphStyle(name='Heading2Custom', fontSize=14, leading=18, spaceAfter=8))
    if 'MyBullet' not in styles:
        styles.add(ParagraphStyle(name='MyBullet', fontSize=11, leading=14, leftIndent=15, bulletIndent=5))

    # Título
    titulo = "Reporte Técnico: Sistema de Detección de Comportamientos Anómalos en Ganado"
    elementos.append(Paragraph(titulo, styles['Title']))
    elementos.append(Spacer(1, 12))
    fecha = f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    elementos.append(Paragraph(fecha, styles['Normal']))
    elementos.append(Spacer(1, 20))

    # Introducción
    intro_content = """
Este documento presenta los resultados completos del proyecto de investigación desarrollado para el 
Magíster en Data Science, cuyo objetivo principal fue diseñar e implementar una metodología robusta 
para el procesamiento y análisis de datos de comportamiento animal mediante sensores IoT.

El proyecto se basó en el dataset público "db-cow-walking-IoT", que registra comportamientos de vacas 
de pastoreo (caminata, pastoreo y reposo) mediante collares equipados con sensores IMU (MPU9250 y BNO055). 
Nuestra contribución amplía este trabajo mediante la generación de nuevas variables derivadas y la aplicación 
de técnicas avanzadas de reducción de dimensionalidad para detectar comportamientos anómalos asociados a 
problemas sanitarios como infestaciones por moscas.
"""
    elementos.append(Paragraph("1. Introducción", styles['Heading2Custom']))
    elementos.append(Paragraph(intro_content, styles['Normal']))
    elementos.append(Spacer(1, 12))

    # Metodología
    metodologia_items = [
        "Carga y unificación de datos desde múltiples archivos CSV organizados por comportamiento",
        "Imputación de valores faltantes utilizando KNNImputer (k=5)",
        "Escalado de variables mediante StandardScaler (media=0, std=1)",
        "Codificación de etiquetas con LabelEncoder",
        "División estratificada de datos (80% entrenamiento, 20% prueba)",
        "Generación de variables derivadas: magnitudes de aceleración y giroscopio, estadísticas móviles",
        "Indicadores comportamentales: 'inquietud_alta' y 'actividad_extrema'",
        "Técnicas de reducción de dimensionalidad: PCA, LDA, t-SNE, UMAP",
        "Evaluación de rendimiento: Accuracy KNN, Silhouette, ARI, NMI, KMeans"
    ]

    elementos.append(Paragraph("2. Metodología", styles['Heading2Custom']))
    for item in metodologia_items:
        elementos.append(Paragraph(f"&bull; {item}", styles['MyBullet']))
    elementos.append(Spacer(1, 12))

    # Resultados - tabla
    elementos.append(Paragraph("3. Resultados", styles['Heading2Custom']))
    data = [
        ["Método", "Accuracy KNN", "Silhouette", "ARI", "NMI"],
        ["PCA", 0.914, 0.115, 0.233, 0.239],
        ["LDA", 0.914, 0.115, 0.233, 0.239],
        ["t-SNE", 0.942, 0.407, 0.029, 0.045],
        ["UMAP", 0.930, 0.442, 0.081, 0.179]
    ]
    table = Table(data, hAlign='LEFT', colWidths=[80, 60, 60, 60, 60])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    elementos.append(table)
    elementos.append(Spacer(1, 12))

    # Ejemplo de gráfico
    elementos.append(Paragraph("3.1 Gráfico de Accuracy", styles['Heading2Custom']))
    fig, ax = plt.subplots(figsize=(5,3))
    métodos = ['PCA','LDA','t-SNE','UMAP']
    accuracies = [0.914,0.914,0.942,0.930]
    ax.bar(métodos, accuracies, color='skyblue')
    ax.set_ylim(0,1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparación de Accuracy por método')
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='PNG')
    plt.close(fig)
    img_buffer.seek(0)
    elementos.append(Image(img_buffer, width=400, height=200))
    elementos.append(Spacer(1, 12))

    # Conclusiones
    conclusiones_items = [
        "Se desarrolló una metodología robusta para el procesamiento de datos de sensores IoT",
        "Las variables derivadas demostraron ser efectivas para capturar patrones comportamentales",
        "Las técnicas no lineales (t-SNE, UMAP) superaron a los métodos lineales en clasificación",
        "Se validó la detección de comportamientos anómalos asociados a problemas sanitarios"
    ]
    elementos.append(Paragraph("4. Conclusiones", styles['Heading2Custom']))
    for item in conclusiones_items:
        elementos.append(Paragraph(f"&bull; {item}", styles['MyBullet']))
    elementos.append(Spacer(1, 12))

    # Trabajo futuro
    futuro_items = [
        "Integración con sistemas de drones para monitoreo a gran escala",
        "Desarrollo de modelos predictivos en tiempo real",
        "Implementación de técnicas de deep learning",
        "Validación en campo con datasets más extensos",
        "Desarrollo de interfaces móviles para acceso remoto"
    ]
    elementos.append(Paragraph("4.1 Trabajo Futuro", styles['Heading2Custom']))
    for item in futuro_items:
        elementos.append(Paragraph(f"&bull; {item}", styles['MyBullet']))

    # Generar PDF
    doc.build(elementos)
    buffer.seek(0)
    return buffer.getvalue()
