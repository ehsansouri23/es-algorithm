import numpy as np
import csv
import pandas as pd
import random


def classification_ystar(output):
    output = [int(x) for x in output]
    classes = np.unique(output)
    c = len(classes)
    l = len(output)
    y_star = np.ones([l, c], dtype=int)
    y_star = np.negative(y_star)

    for i in range(0, l):
        if output[i] == -1:
            y_star[i][1] = 1
        else:
            y_star[i][output[i] - 1] = 1

    return y_star


def regression_ystar(output):
    return output

#
# train_data = []
# test_data = []
train_data_address = 'RBF/4clstrain1200.csv'
test_data_address = 'RBF/test40000.csv'

# with open(train_data_address, newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         train_data.append(row)

train_data = pd.read_csv(train_data_address,header=0)
# train_data = [[float(x) for x in row] for row in train_data]
# train_data = np.array(train_data)

train_data = train_data.to_numpy()
# with open(test_data_address, newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         test_data.append(row)

test_data = pd.read_csv(test_data_address,header=0)
# test_data = [[float(x) for x in row] for row in test_data]
# test_data = np.array(test_data)
test_data = test_data.to_numpy()

train_output = train_data[:, train_data.shape[1] - 1]
train_data = np.delete(train_data, train_data.shape[1] - 1, axis=1)

# test_output = test_data[:, test_data.shape[1] - 1]
# test_data = np.delete(test_data, test_data.shape[1] - 1, axis=1)

train_cn = 1 / np.max(train_data)
test_cn = 1 / np.max(test_data)

train_data *= train_cn
test_data *= test_cn

min_data = np.min(train_data)
max_data = np.max(train_data)

problems = {'classification': classification_ystar, 'regression': regression_ystar}

print('Enter Problem type :')
problem = input()

train_ystar = problems[problem](train_output)
# test_ystar = problems[problem](test_output)
