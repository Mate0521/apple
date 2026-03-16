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

    # convertir calidad a números
    df["Quality"] = df["Quality"].map({
        "good": 1,
        "bad": 0
    })

    # EJECUTAR EDA
    eda_head, eda_stats, eda_quality = create_eda(df)

    # PREPARAR DATOS (Tratar de ajustar correctamenta el 80/20)
    X = df.drop("Quality", axis=1).values.tolist()
    y = df["Quality"].tolist()

    data = list(zip(X, y))

    random.shuffle(data)

    split = int(len(data) * 0.8)

    train = data[:split]
    test = data[split:]

    X_train = [d[0] for d in train]
    y_train = [d[1] for d in train]

    X_test = [d[0] for d in test]
    y_test = [d[1] for d in test]

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
    nn.train(X_train, y_train, epochs=1000, lr=0.01)

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


# ENTRENAR MODELO
accuracy = train_model()

# PÁGINA WEB
@app.route("/")
def home():

    return f"""
    <h1>Apple Quality Neural Network</h1>

    <h2>Resultado del modelo</h2>
    <p><b>Accuracy:</b> {accuracy}</p>

    <h2>Primeras filas del dataset</h2>
    {eda_head}

    <h2>Estadísticas del dataset</h2>
    {eda_stats}

    <h2>Distribución de calidad</h2>
    {eda_quality}

    <h2>Relación entre Sweetness y Quality</h2>
    <img src="/static/sweetness_vs_quality.png" width="600">

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

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)