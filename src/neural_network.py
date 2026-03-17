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
    def train(self, X, y, epochs, lr, X_test=None, y_test=None):

        # Lista donde guardaremos el accuracy en cada epoch
        # Esto permitirá graficar el comportamiento del modelo
        accuracy_history = []

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


                # Diferencia entre el valor real y la predicción
                # Este error indica qué tan lejos está la red de la respuesta correcta
                error = y_true - output

                # Derivada de la función sigmoide
                # Permite saber cómo ajustar la salida
                # Teóricamente: gradiente del error respecto a la salida
                d_output = error * sigmoid_derivative(output)

                # Ajustamos los pesos de la capa oculta → salida
                for j in range(self.hidden_size):

                    # Regla de actualización (Gradient Descent):
                    # w = w + (learning_rate * gradiente * activación)
                    self.w2[j] += lr * d_output * hidden[j]


                # Ajustamos los pesos de entrada → capa oculta
                for i in range(self.input_size):

                    for j in range(self.hidden_size):

                        # Gradiente de la neurona oculta
                        # Se propaga el error hacia atrás (backpropagation)
                        d_hidden = d_output * self.w2[j] * sigmoid_derivative(hidden[j])

                        # Actualización del peso
                        self.w1[i][j] += lr * d_hidden * x[i]

            # Solo calculamos accuracy si se proporcionan datos de test
            if X_test is not None and y_test is not None:

                correct = 0

                # Evaluamos el modelo con datos no vistos
                for x_val, y_val in zip(X_test, y_test):

                    pred = self.predict(x_val)

                    # Contamos predicciones correctas
                    if pred == y_val:
                        correct += 1

                # Accuracy = aciertos / total
                accuracy = correct / len(X_test)

                # Guardamos el valor para la gráfica
                accuracy_history.append(accuracy)

                # (Opcional) imprimir progreso
                #print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.4f}")

        # Retornamos el historial completo
        return accuracy_history

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