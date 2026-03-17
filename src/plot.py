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