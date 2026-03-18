# Importamos librerías necesarias
# matplotlib -> creación de gráficas
# seaborn -> gráficas estadísticas
# os -> permite obtener variables del sistema como el puerto

import matplotlib.pyplot as plt
import seaborn as sns


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



# 1. KDE 
    plt.figure()
    sns.kdeplot(data=df, x="Size", hue="Quality", fill=True)
    plt.title("Densidad de Size por Quality")
    plt.savefig("src/static/kde_size.png")
    plt.close()

    # 2. Histograma normalizado
    plt.figure()
    sns.histplot(data=df, x="Sweetness", hue="Quality", stat="density", kde=True)
    plt.title("Distribución normalizada de Sweetness")
    plt.savefig("src/static/hist_sweetness_norm.png")
    plt.close()

    # 3. Scatter (relación entre variables)
    plt.figure()
    sns.scatterplot(data=df, x="Weight", y="Size", hue="Quality")
    plt.title("Relación Weight vs Size")
    plt.savefig("src/static/scatter_weight_size.png")
    plt.close()

    # 4. Promedios por clase 
    plt.figure()
    df.groupby("Quality").mean().plot(kind="bar")
    plt.title("Promedio de variables por Quality")
    plt.savefig("src/static/mean_variables.png")
    plt.close()

    # 5. Heatmap de correlación
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlación entre variables")
    plt.savefig("src/static/correlation.png")
    plt.close()



    return head, stats, quality

