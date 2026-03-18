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
from plot import plot_accuracy, plot_architecture


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

    # Se crea el diagrama de la arquitectura de la red neuronal
    plot_architecture(input_size, hidden_size)

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

    <h2>📊 Análisis Exploratorio de Datos (EDA)</h2>

    <h3>1. Densidad de Size</h3>
    <img src="/static/kde_size.png" width="500">
    <p>
    Esta gráfica muestra la distribución del tamaño de las manzanas según su calidad.
    Se observa que las manzanas buenas tienden a tener mayor tamaño, aunque existe superposición.
    </p>

    <h3>2. Distribución de Sweetness</h3>
    <img src="/static/hist_sweetness_norm.png" width="500">
    <p>
    Se analiza la dulzura de las manzanas de forma normalizada.
    Permite comparar proporciones reales entre clases sin depender de la cantidad de datos.
    </p>

    <h3>3. Relación Weight vs Size</h3>
    <img src="/static/scatter_weight_size.png" width="500">
    <p>
    Muestra la relación entre peso y tamaño.
    No hay una separación clara entre clases, lo que indica que se necesitan múltiples variables.
    </p>

    <h3>4. Promedio de variables por Quality</h3>
    <img src="/static/mean_variables.png" width="500">
    <p>
    Permite comparar los valores promedio de cada variable entre manzanas buenas y malas.
    Ayuda a identificar cuáles características influyen más en la calidad.
    </p>

    <h3>5. Correlación entre variables</h3>
    <img src="/static/correlation.png" width="500">
    <p>
    El mapa de calor muestra la relación entre variables.
    Permite identificar cuáles tienen mayor influencia sobre la calidad.
    </p>

    <h2>🤓 Conclusión</h2>
    <p>
    Teniendo en cuenta los datos analizados se puede concluir que no existe un punto de 
    comparación único y específico que pueda determinar en su totalidad si una manzana es 
    buena o mala, se puede observar que para obtener el resultado deseado hay que tener en 
    cuenta muchas variables y relacionarlas entre sí tal como se muestra en los gráficos. 
    En su mayoría se puede decir que si una manzana es grande, con altos grados de dulzura y 
    jugosidad, y bajo grado de madurez, podría estar en el intervalo de ser una buena manzana.
    </p>


    <h2>Arquitectura de la red neuronal</h2>
    <img src="/static/architecture.png" width="600">

    <h2>Algoritmo</h2>

    <p>
    Se implementó una red neuronal entrenada con el algoritmo de <b>Backpropagation</b>, el cual ajusta los pesos de la red en función del error de predicción.
    </p>

    <h3>1. Función de activación (Sigmoid)</h3>

    <pre>
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    </pre>

    <p>
    Esta función transforma cualquier valor en un rango entre 0 y 1, lo que permite interpretar la salida como una probabilidad en un problema de clasificación binaria.
    </p>

    <h3>2. Inicialización de pesos</h3>

    <pre>
    self.w1 = [[random.uniform(-1,1) for _ in range(hidden_size)]
            for _ in range(input_size)]

    self.w2 = [random.uniform(-1,1) for _ in range(hidden_size)]
    </pre>

    <p>
    Los pesos se inicializan aleatoriamente para que la red comience a aprender desde cero. 
    w1 conecta la capa de entrada con la capa oculta, mientras que w2 conecta la capa oculta con la salida.
    </p>

    <h3>3. Propagación hacia adelante (Forward)</h3>

    <pre>
    s += x[i] * self.w1[i][j]
    hidden.append(sigmoid(s))
    </pre>

    <p>
    Cada entrada se multiplica por su peso correspondiente y se suma. Luego se aplica la función sigmoide para obtener la activación de cada neurona en la capa oculta.
    </p>

    <pre>
    s += hidden[j] * self.w2[j]
    output = sigmoid(s)
    </pre>

    <p>
    Las salidas de la capa oculta se combinan para generar la salida final de la red, que representa la probabilidad de la clase.
    </p>

    <h3>4. Cálculo del error</h3>

    <pre>
    error = y_true - output
    </pre>

    <p>
    Se calcula la diferencia entre el valor real y la predicción del modelo. Este error indica qué tan precisa fue la predicción.
    </p>

    <h3>5. Backpropagation</h3>

    <pre>
    d_output = error * sigmoid_derivative(output)
    </pre>

    <p>
    Se calcula el gradiente del error utilizando la derivada de la función sigmoide, lo que permite determinar cómo ajustar los pesos.
    </p>

    <h3>6. Actualización de pesos</h3>

    <pre>
    self.w2[j] += lr * d_output * hidden[j]
    </pre>

    <p>
    Se ajustan los pesos entre la capa oculta y la salida en función del error y la contribución de cada neurona.
    </p>

    <pre>
    d_hidden = d_output * self.w2[j] * sigmoid_derivative(hidden[j])
    self.w1[i][j] += lr * d_hidden * x[i]
    </pre>

    <p>
    Se propaga el error hacia atrás y se ajustan los pesos entre la capa de entrada y la capa oculta.
    </p>

    <h3>7. Predicción</h3>

    <pre>
    if output >= 0.5:
        return 1
    else:
        return 0
    </pre>

    <p>
    Se define una regla de decisión: si la probabilidad es mayor o igual a 0.5, se clasifica como "buena calidad", de lo contrario como "mala calidad".
    </p>

    """

# EJECUTAR SERVIDOR
if __name__ == "__main__":

    os.makedirs("src/static", exist_ok=True)

    accuracy = train_model()

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)