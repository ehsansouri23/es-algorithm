import array
import random
import Problem as p
import RBF_ANN as rbf
import plot as plt
import numpy as np
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import time
import matplotlib.pyplot as pl

n = p.train_data.shape[1]
m = rbf.m  # p.train_ystar.shape[1]
IND_SIZE = m * (n + 1)
MIN_VALUE = p.min_data
MAX_VALUE = p.max_data
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3

creator.create("FitnessMin", base.Fitness, weights=[-1.0])
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")


# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


def evaluate(individual):
    ind = np.array(individual)
    ind = ind.reshape(m, n + 1)
    y = rbf.network(p.train_data, p.train_ystar, ind)
    fitness = rbf.network_error(y, p.train_ystar)
    return [fitness]


# gamas = ind[:, n]
# vs = np.delete(ind, n, axis=1)
# G = rbf.g_matrix(p.train_data, vs, gamas)
# W = rbf.weights(G, p.y_star)
# y = rbf.y(G, W)
# fitness = rbf.classification_error(y, p.y_star)


toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def main():
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=25, stats=stats, halloffame=hof)

    vs, gamas,w = rbf.validation(p.train_data, p.train_ystar, pop)
    answer = rbf.test(p.test_data,w, vs, gamas)
    plt.plot(p.test_data, p.test_cn, vs, gamas, answer)
    return pop, logbook, hof


if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)
