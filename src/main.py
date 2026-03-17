# IMPORTACIÓN DE LIBRERÍAS
# Flask permite crear una pequeña aplicación web
from flask import Flask

# pandas se usa para cargar el dataset
import pandas as pd

# random permite mezclar los datos antes del entrenamiento
import random

# os permite obtener variables del sistema como el puerto
import os

# importar nuestra red neuronal
from neural_network import NeuralNetwork

# importar el análisis exploratorio
from eda import create_eda

# importar función para graficar el accuracy
from plot import plot_accuracy


# CREAR APP WEB
app = Flask(__name__)

# variables globales para mostrar en la web
eda_head = ""
eda_stats = ""
eda_quality = ""

# FUNCIÓN DE ENTRENAMIENTO
def train_model():

    global eda_head, eda_stats, eda_quality

    # CARGAR DATASET
    df = pd.read_csv("data/apple_quality.csv")

    # eliminar columna id
    if "A_id" in df.columns:
        df = df.drop("A_id", axis=1)

  
    # LIMPIEZA DE DATOS
    for col in df.columns:
        if col != "Quality":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # NORMALIZACIÓN DE DATOS
    for col in df.columns:
        if col != "Quality":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)

    

    # convertir calidad a números
    df["Quality"] = df["Quality"].map({
        "good": 1,
        "bad": 0
    })

    # EJECUTAR EDA
    eda_head, eda_stats, eda_quality = create_eda(df)

    # PREPARAR DATOS 80/20
    # Separar variables de entrada (X) y variable objetivo (y)
    # X contiene todas las características de las manzanas
    # y contiene la calidad (0 = bad, 1 = good)
    X = df.drop("Quality", axis=1).values.tolist()
    y = df["Quality"].tolist()

    # Unimos X - y en una sola estructura
    # Esto facilita mezclar los datos sin perder la relación entre entrada y salida
    data = list(zip(X, y))

    # Mezclamos los datos aleatoriamente
    # Esto evita que el modelo aprenda patrones incorrectos por el orden del dataset
    random.seed(42)
    random.shuffle(data)

    # Calculamos el punto de división para el 80% de los datos
    # Se utilizará para separar entrenamiento y prueba
    split = int(len(data) * 0.8)

    # Dividimos los datos:
    # 80% para entrenamiento
    # 20% para evaluación (prueba)
    train = data[:split]
    test = data[split:]

    # Separamos nuevamente las entradas (X) y salidas (y) para entrenamiento
    X_train = [d[0] for d in train] # Características 
    y_train = [d[1] for d in train] # Etiquetas 

    # Separamos entradas y salidas para prueba
    X_test = [d[0] for d in test]
    y_test = [d[1] for d in test]

    # Mostramos información en consola para verificar la correcta división
    print(f"Total datos: {len(data)}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # CREAR RED NEURONAL
    # input_size representa cuántas características tiene cada manzana
    # Tomamos la primera fila de X_train y contamos cuántos valores tiene
    input_size = len(X_train[0])

    # hidden_size indica cuántas neuronas tendrá la capa oculta
    # Este valor se elige como hiperparámetro del modelo
    hidden_size = 6

    # Creamos la red neuronal usando la clase NeuralNetwork
    # Se le pasa el número de entradas y el número de neuronas ocultas
    nn = NeuralNetwork(input_size, hidden_size)

    # Mensaje informativo para indicar que el entrenamiento comienza
    print("Entrenando red neuronal...")

    # Entrenamos la red neuronal
    # X_train → datos de entrada
    # y_train → respuestas correctas
    # epochs → número de veces que el modelo verá todo el dataset
    # lr → learning rate (tasa de aprendizaje)
    accuracy_history = nn.train(
        X_train,
        y_train,
        epochs=1000,
        lr=0.01,
        X_test=X_test,
        y_test=y_test
    )

    plot_accuracy(accuracy_history)

    # EVALUACIÓN
    # Variable para contar cuántas predicciones del modelo fueron correctas
    correct = 0

    # Recorremos los datos de prueba (los que NO se usaron para entrenar)
    # zip(X_test, y_test) une cada entrada con su valor real
    for x, y_true in zip(X_test, y_test):

        # La red neuronal realiza una predicción usando la función predict()
        prediction = nn.predict(x)

        # Comparamos la predicción con el valor real
        # Si coinciden significa que el modelo acertó
        if prediction == y_true:
            correct += 1

    #(Tratar de demostrar el accuracy ya sea por una grafica o un metodo de qeu es correcto)
    # Calculamos el accuracy del modelo
    # accuracy = predicciones correctas / total de datos evaluados
    accuracy = correct / len(X_test)

    # Retornamos el accuracy obtenido
    # Este valor indica qué tan bien funciona la red neuronal
    return accuracy

accuracy = None

# PÁGINA WEB
@app.route("/")
def home():

    return f"""
    <h1>Apple Quality Neural Network</h1>

    <h2>Gráfica de Accuracy</h2>
    <img src="/static/accuracy.png" width="600">

    <h2>Resultado del modelo</h2>
    <p><b>Accuracy:</b> {accuracy}</p>

    <h2>Primeras filas del dataset</h2>
    {eda_head}

    <h2>Estadísticas del dataset</h2>
    {eda_stats}

    <h2>Distribución de calidad</h2>
    {eda_quality}

    <h2>Arquitectura de la red neuronal</h2>
    <p>
    Entradas → Capa oculta (6 neuronas) → Salida (1 neurona)
    </p>

    <h2>Algoritmo</h2>
    <p>
    Se utilizó Backpropagation para entrenar la red neuronal
    ajustando los pesos según el error de predicción.
    </p>
    """

# EJECUTAR SERVIDOR
if __name__ == "__main__":

    accuracy = train_model()


    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)