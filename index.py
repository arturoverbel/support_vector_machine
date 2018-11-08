import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

FEATURE1 = 'SepalWidthCm'
FEATURE2 = 'PetalWidthCm'

FEATURE_TARGET = 'Species'
CLASS_A = 'Iris-setosa'
CLASS_B = 'Iris-versicolor'

FILE_NAME = 'iris.csv'

#Set data from file
df = pd.read_csv(FILE_NAME)
target = df[FEATURE_TARGET]

x = df[FEATURE1]
y = df[FEATURE2]

df = df.drop([FEATURE1, FEATURE2], axis=1)
Y = []
X = []
classA_x = []
classA_y = []
classB_x = []
classB_y = []

for idx, val in enumerate(target):
    is_data_set = False
    if val == CLASS_A:
        Y.append(-1)
        classA_x.append(x[idx])
        classA_y.append(y[idx])
        is_data_set = True
    elif val == CLASS_B:
        Y.append(1)
        classB_x.append(x[idx])
        classB_y.append(y[idx])
        is_data_set = True

    if is_data_set:
        X.append([x[idx], y[idx]])

## Shuffle and split the data into training and test set
X, Y = shuffle(X, Y)
x_train = []
y_train = []
x_test = []
y_test = []

porcent = 90

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=porcent/100)

data_train = np.array(x_train) #Example data
label_train = np.array(y_train) #Label
data_test = np.array(x_test)
label_test = np.array(y_test)

label_train = label_train.reshape(porcent, 1)
label_test = label_test.reshape(100-porcent, 1)

data_train_x = data_train[:, 0]
data_train_y = data_train[:, 1]

data_train_x = data_train_x.reshape(porcent, 1)
data_train_y = data_train_y.reshape(porcent, 1)

w1 = np.zeros((porcent, 1))
w2 = np.zeros((porcent, 1))

epochs = 1
alpha = 0.0001

while (epochs < 10000):
    y = w1 * data_train_x + w2 * data_train_y
    prod = y * label_train
    count = 0
    for val in prod:
        if (val >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)

        else:
            cost = 1 - val
            w1 = w1 + alpha * (data_train_x[count] * label_train[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (data_train_y[count] * label_train[count] - 2 * 1 / epochs * w2)
        count += 1
    epochs += 1

index = list(range(100-porcent, porcent))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

print(w1)
print(w2)

w1 = w1.reshape(100-porcent, 1)
w2 = w2.reshape(100-porcent, 1)

data_test_x = data_test[:, 0]
data_test_y = data_test[:, 1]

data_test_x = data_test_x.reshape(100-porcent, 1)
data_test_y = data_test_y.reshape(100-porcent, 1)

## Predict
y_pred = w1 * data_test_x + w2 * data_test_y
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print("Precisión: ", accuracy_score(y_test, predictions) * 100, "%")

plt.figure(figsize=(8, 6))
plt.scatter(classA_x, classA_y, marker='+', color='blue')
plt.scatter(classB_x, classB_y, marker='_', color='red')
plt.show()
