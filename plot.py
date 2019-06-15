import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pc


def plot_classification(data, vs, gamas, y):
    t = np.argmax(y, axis=1) - np.argmax(y_star, axis=1)
    a = np.where(t == 0)
    b = np.where(t != 0)
    correct = np.array(data)[a]
    wrong = np.array(data)[b]
    plt.scatter(correct[:, 0], correct[:, 1], c='g', marker='.')
    plt.scatter(wrong[:, 0], wrong[:, 1], c='r', marker='.')
    plt.scatter(vs[:, 0], vs[:, 1], c='y', marker='.')
    ax = plt.gca()
    for i in range(0, len(vs)):
        c = plt.Circle(tuple(vs[i]), 1 / (gamas[i] ** 0.5), facecolor='none', edgecolor='black')
        ax.add_patch(c)
        plt.axis('scaled')
    plt.show()


def plot_regression(data, y, y_star):
    x = [i for i in range(0, len(data))]
    plt.plot(x, y, c='r', marker='.')
    plt.plot(x, y_star, c='g', marker='.')
    plt.show()


# def plot(data, vs, gamas, y, y_star):
def plot(data, vs, gamas, y, y_star):
    # data = (1 / cn) * data
    # vs = (1 / cn) * vs
    # if len(y.shape) == 1:
    #     plot_regression(data, y, y_star)
    # else:
        plot_classification(data, vs, gamas, y)
