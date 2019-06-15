import numpy as np

m = 7


def g(k, x, v):
    d = np.dot((x - v), (x - v))
    return np.e ** (-k * d)


def g_matrix(xs, vs, ks):
    m = len(vs)
    l = len(xs)
    G = [[g(ks[j], xs[i], vs[j]) for j in range(0, m)] for i in range(0, l)]
    G = np.array(G)
    return G


def y(g_matrix, weights):
    y = np.matmul(g_matrix, weights)
    return y


def weights(g_matrix, y_star):
    m = g_matrix.shape[1]
    gt = np.transpose(g_matrix)
    gtg = np.matmul(gt, g_matrix)
    I = np.identity(m) * 2
    inv = np.linalg.inv(np.add(gtg, I))
    weights = np.matmul(np.matmul(inv, gt), y_star)
    return weights


def classification_error(y, y_star):
    t = np.argmax(y, axis=1) - np.argmax(y_star, axis=1)
    error = np.sum(np.sign(np.abs(t))) / len(y)
    return error


def regression_error(y, y_star):
    a = np.subtract(y, y_star)
    at = np.transpose(a)
    error = 0.5 * np.matmul(at, a)
    return error


def network(xs, y_star, ind):
    n = ind.shape[1] - 1
    # m = ind.shape[0]
    gamas = ind[:, n]
    vs = np.delete(ind, n, axis=1)
    G = g_matrix(xs, vs, gamas)
    W = weights(G, y_star)
    return y(G, W),W


def network_error(y, y_star):
    if len(y.shape) == 1:
        return regression_error(y, y_star)
    else:
        return classification_error(y, y_star)


def validation(data, y_star, pop):
    errors = []
    n = data.shape[1]
    for ind in pop:
        ind = np.array(ind)
        ind = ind.reshape(m, n + 1)
        y,w = network(data, y_star, ind)
        error = network_error(y, y_star)
        errors.append(error)
    i = errors.index(min(errors))
    print(min(errors))
    ind = pop[i]
    y, w = network(data, y_star, ind)
    ind = np.array(ind)
    ind = ind.reshape(m, n + 1)
    gamas = ind[:, n]
    vs = np.delete(ind, n, axis=1)
    return vs, gamas,w


def test(data, w, vs, gamas):
    G = g_matrix(data, vs, gamas)
    # if len(y_star.shape) != 1:
    #     print(1 - classification_error(y(G, W), y_star))
    return y(G, w)
