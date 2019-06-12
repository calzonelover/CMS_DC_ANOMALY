import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(x, loss_train, loss_valid, title):
    """Plots the training and validation loss"""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(x, loss_train, '-b', label='Training')
    plt.plot(x, loss_valid, '-r', linestyle=(0, (1, 2)), label='Validation')
    plt.legend(["Training", "Validation"], loc="upper right", frameon=False)
    plt.yscale("log")
    # plt.show()
    plt.savefig('{}.png'.format(title))

if __name__ == "__main__":
    for i in range(1,6):
        data = pd.read_csv('logs/sparse model{}.txt'.format(i), sep=" ")
        # print(data['EP'].shape)
        x = data['EP']
        loss_train = data['loss_test']
        loss_valid = data['loss_valid']
        plot_loss(x, loss_train, loss_valid, title="Sparse Model{}".format(i))