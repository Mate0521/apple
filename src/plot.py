import matplotlib.pyplot as plt


def plot_accuracy(accuracy_history):

    epochs = list(range(1, len(accuracy_history)+1))

    plt.figure()

    plt.plot(epochs, accuracy_history)

    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.savefig("src/static/accuracy.png")


def plot_architecture(input_size, hidden_size, output_size=1):

    plt.figure()

    # Posiciones en eje X
    x_input = 0
    x_hidden = 1
    x_output = 2

    # Dibujar neuronas de entrada
    for i in range(input_size):
        plt.scatter(x_input, i)
        plt.text(x_input, i, f"In {i+1}", fontsize=8)

    # Dibujar capa oculta
    for j in range(hidden_size):
        plt.scatter(x_hidden, j)
        plt.text(x_hidden, j, f"H {j+1}\n(sigmoid)", fontsize=8)

    # Dibujar salida
    for k in range(output_size):
        plt.scatter(x_output, k)
        plt.text(x_output, k, f"Out\n(sigmoid)", fontsize=8)

    # Dibujar conexiones
    for i in range(input_size):
        for j in range(hidden_size):
            plt.plot([x_input, x_hidden], [i, j])

    for j in range(hidden_size):
        for k in range(output_size):
            plt.plot([x_hidden, x_output], [j, k])

    plt.title("Arquitectura de la Red Neuronal")

    plt.axis('off')

    plt.savefig("src/static/architecture.png")