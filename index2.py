import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

num_samples=10
num_features=2

samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
labels = 2 * (samples.sum(axis=1) > 0) - 1.0

point_x = samples[:, 0]
point_y = samples[:, 1]

labelss = np.hstack(labels)

w1 = np.zeros((num_samples, 1))
w2 = np.zeros((num_samples, 1))

epochs = 1
alpha = 0.0001

print(w1)
print(point_x)
print(w1 * point_x)

while (epochs < 10000):
    y = w1 * point_x + w2 * point_y
    prod = y * labels
    #print(epochs)
    count = 0
    for val in prod:
        if (val >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)

        else:
            cost = 1 - val
            print(point_x)
            print(labelss)
            a = point_x[count] * labelss[count]
            w1 = w1 + alpha * (point_x[count] * labelss[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (point_y[count] * labelss[count] - 2 * 1 / epochs * w2)
        count += 1
    epochs += 1