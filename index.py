import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

FEATURE1 = 'SepalWidthCm'
FEATURE2 = 'PetalLengthCm'

FEATURE_TARGET = 'Species'
CLASS_A = 'Iris-setosa'
CLASS_B = 'Iris-versicolor'

FILE_NAME = 'iris_missing_values.csv'

# Set data from file
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

min_x_plot = np.inf
max_x_plot = 0

classA_missing = []
classB_missing = []

for idx, val in enumerate(target):
    is_data_set = False

    if x[idx] == '?' or y[idx] == '?':
        print('Missing data: (', x[idx], ', ', y[idx], ', ', val, ')')
        if val == CLASS_A:
            classA_missing.append([x[idx], y[idx]])
        elif val == CLASS_B:
            classB_missing.append([x[idx], y[idx]])
        continue

    data_x = float(x[idx])
    data_y = float(y[idx])
    if val == CLASS_A:
        classA_x.append(data_x)
        classA_y.append(data_y)
        is_data_set = True
        if data_x < min_x_plot:
            min_x_plot = data_x
        if data_x > max_x_plot:
            max_x_plot = data_x
        Y.append(-1)
    elif val == CLASS_B:
        classB_x.append(data_x)
        classB_y.append(data_y)
        is_data_set = True
        if data_x < min_x_plot:
            min_x_plot = data_x
        if data_x > max_x_plot:
            max_x_plot = data_x
        Y.append(1)

    if is_data_set:
        X.append([data_x, data_y])

print(classA_missing)
if len(classA_missing) > 0:
    for data_missing in classA_missing:
        if data_missing[0] == '?':
            data_missing[0] = np.mean(classA_x)
            print(data_missing[0])
        else:
            data_missing[1] = np.mean(classA_y)
            print(data_missing[1])

        classA_x.append(data_missing[0])
        classA_y.append(data_missing[1])
        X.append([data_missing[0], data_missing[1]])
        Y.append(1)

    for data_missing in classB_missing:
        if data_missing[0] == '?':
            data_missing[0] = np.mean(classB_x)
            print(data_missing[0])
        else:
            data_missing[1] = np.mean(classB_y)
            print(data_missing[1])

        classB_x.append(data_missing[0])
        classB_y.append(data_missing[1])
        X.append([data_missing[0], data_missing[1]])
        Y.append(-1)

print(classB_missing)

# Shuffle and split the data into training and test set
X, Y = shuffle(X, Y)
x_train = []
y_train = []
x_test = []
y_test = []

porcent = 90

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=porcent/100)

len_total = len(X)
len_dataset_train = len(x_train)

# Example data
data_train = np.array(x_train)
# Label
label_train = np.array(y_train)
data_test = np.array(x_test)
label_test = np.array(y_test)

label_train = label_train.reshape(len_dataset_train, 1)
label_test = label_test.reshape(len_total-len_dataset_train, 1)

data_train_x = data_train[:, 0]
data_train_y = data_train[:, 1]

data_train_x = data_train_x.reshape(len_dataset_train, 1)
data_train_y = data_train_y.reshape(len_dataset_train, 1)

w1 = np.zeros((len_dataset_train, 1))
w2 = np.zeros((len_dataset_train, 1))

epochs = 1
alpha = 0.0001

while epochs < 10000:
    y = (w1 * data_train_x) + (w2 * data_train_y)
    prod = y * label_train
    count = 0
    for val in prod:
        if val >= 1:
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)

        else:
            cost = 1 - val
            w1 = w1 + alpha * (data_train_x[count] * label_train[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (data_train_y[count] * label_train[count] - 2 * 1 / epochs * w2)
        count += 1
    epochs += 1

index = list(range(len_total-len_dataset_train, len_dataset_train))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

w1 = w1.reshape(len_total-len_dataset_train, 1)
w2 = w2.reshape(len_total-len_dataset_train, 1)

data_test_x = data_test[:, 0]
data_test_y = data_test[:, 1]

data_test_x = data_test_x.reshape(len_total-len_dataset_train, 1)
data_test_y = data_test_y.reshape(len_total-len_dataset_train, 1)

## Predict
y_pred = w1 * data_test_x + w2 * data_test_y
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print("Precisión: ", accuracy_score(y_test, predictions) * 100, "%")

since = int(min_x_plot)-1
until = int(max_x_plot)+2
range_to_plot = range(since, until)

# 45,5.1,3.8,1.9,0.4,Iris-setosa


def func_lineal(a, b, xx):
    return ((a*-1.0) / b) * xx


plt.figure(figsize=(8, 6))
plt.scatter(classA_x, classA_y, marker='+', color='blue')
plt.scatter(classB_x, classB_y, marker='_', color='red')
plt.plot(range_to_plot, [func_lineal(w1[0], w2[0], i) for i in range_to_plot])
plt.show()
