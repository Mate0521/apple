import random
import math


# FUNCIÓN DE ACTIVACIÓN

# La función sigmoid convierte cualquier número
# en un valor entre 0 y 1.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivada de sigmoid
# Necesaria para el algoritmo de aprendizaje
def sigmoid_derivative(x):
    return x * (1 - x)


# CLASE RED NEURONAL
class NeuralNetwork:

    def __init__(self, input_size, hidden_size):

        # número de neuronas de entrada
        self.input_size = input_size

        # número de neuronas en capa oculta
        self.hidden_size = hidden_size

        # Pesos entre capa de entrada y capa oculta
        # Inicializados con valores aleatorios
        self.w1 = [[random.uniform(-1,1) for _ in range(hidden_size)]
                   for _ in range(input_size)]

        # Pesos entre capa oculta y salida
        self.w2 = [random.uniform(-1,1) for _ in range(hidden_size)]


   
    # PROPAGACIÓN HACIA ADELANTE
    def forward(self, x):

        # Lista donde se guardarán las salidas
        # de cada neurona de la capa oculta
        hidden = []

        # CÁLCULO DE LA CAPA OCULTA
        # Recorremos cada neurona de la capa oculta
        for j in range(self.hidden_size):

            # Variable acumuladora
            # guardará la suma ponderada
            s = 0

            # Recorremos cada variable de entrada
            for i in range(self.input_size):

                # Multiplicamos cada entrada por su peso correspondiente
                # y sumamos el resultado
                s += x[i] * self.w1[i][j]

            # Aplicamos la función de activación sigmoide
            # para introducir no linealidad en la red
            hidden.append(sigmoid(s))


        # CÁLCULO DE LA SALIDA FINAL
        s = 0

        # Sumamos las salidas de la capa oculta
        # multiplicadas por sus pesos hacia la salida
        for j in range(self.hidden_size):

            s += hidden[j] * self.w2[j]

        # Aplicamos nuevamente la función sigmoide
        # para obtener un valor entre 0 y 1
        output = sigmoid(s)

        # Retornamos
        # hidden -> activaciones de la capa oculta
        # output -> predicción final de la red
        return hidden, output


    # ENTRENAMIENTO (BACKPROPAGATION)
    def train(self, X, y, epochs, lr):

        # Repetimos el entrenamiento varias veces
        # Cada repetición completa sobre todos los datos se llama "epoch"
        for epoch in range(epochs):

            # Recorremos cada ejemplo del dataset
            # x -> variables de entrada
            # y_true -> valor real 
            for x, y_true in zip(X, y):

                # Calculamos la salida de la red neuronal
                # hidden -> valores de la capa oculta
                # output -> predicción final de la red
                hidden, output = self.forward(x)
           
                # Calculamos la diferencia entre el valor real
                # y la predicción de la red neuronal
                error = y_true - output

                # Derivada de la función sigmoide
                # Esto indica qué tanto debe cambiar la salida
                d_output = error * sigmoid_derivative(output)

                # Ajustamos los pesos que conectan la capa oculta con la salida
                for j in range(self.hidden_size):

                    # Regla de actualización:
                    # peso = peso + (learning_rate * gradiente * valor_neurona_oculta)
                    self.w2[j] += lr * d_output * hidden[j]

                # ACTUALIZACIÓN PESOS
                # ENTRADA → CAPA OCULTA
                for i in range(self.input_size):

                    for j in range(self.hidden_size):

                        # Calculamos el gradiente para la neurona oculta
                        # usando backpropagation
                        d_hidden = d_output * self.w2[j] * sigmoid_derivative(hidden[j])

                        # Ajustamos el peso correspondiente
                        self.w1[i][j] += lr * d_hidden * x[i]

    # PREDICCIÓN
    def predict(self, x):

        # Ejecutamos la propagación hacia adelante (forward pass)
        # Esto calcula las activaciones de la capa oculta y la salida final
        _, output = self.forward(x)

        # Aplicamos una regla de decisión para clasificar el resultado
        # Si la probabilidad es mayor o igual a 0.5 -> clase 1 (good)
        if output >= 0.5:
            return 1

        # Si la probabilidad es menor a 0.5 -> clase 0 (bad)
        else:
            return 0