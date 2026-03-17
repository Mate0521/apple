# Importamos librerías necesarias
# matplotlib -> creación de gráficas
# seaborn -> gráficas estadísticas
# os -> permite obtener variables del sistema como el puerto

import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("src/static", exist_ok=True)


# FUNCIÓN DE ANÁLISIS EXPLORATORIO
def create_eda(df):

    # TABLAS PARA MOSTRAR
    # Obtener las primeras filas del dataset
    # df.head() muestra los primeros registros de la tabla
    head = df.head().to_html()

    # Obtener estadísticas descriptivas del dataset
    # df.describe() calcula medidas estadísticas como:
    # media, desviación estándar, mínimo, máximo y percentiles
    stats = df.describe().to_html()

    # Obtener la distribución de la variable Quality
    # value_counts() cuenta cuántas manzanas son de cada tipo (good o bad)
    # to_frame() convierte el resultado en una tabla
    quality = df["Quality"].value_counts().to_frame().to_html()


    return head, stats, quality