import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('iris.csv')
df = df.drop(['Id'],axis=1)
target = df['Species']
s = set()
for val in target:
    s.add(val)
s = list(s)
rows = list(range(100, 150))
df = df.drop(df.index[rows])

x = df['SepalLengthCm']
y = df['PetalLengthCm']

setosa_x = x[:50]
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = y[50:]

#plt.figure(figsize=(8,6))
#plt.scatter(setosa_x,setosa_y,marker='+',color='green')
#plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')
#plt.show()

df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
Y = []
target = df['Species']
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)
df = df.drop(['Species'],axis=1)
X = df.values.tolist()
## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

porcent = 10

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=porcent/100)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(porcent, 1)
y_test = y_test.reshape(100-porcent, 1)

train_f1 = x_train[:, 0]
train_f2 = x_train[:, 1]

train_f1 = train_f1.reshape(porcent, 1)
train_f2 = train_f2.reshape(porcent, 1)

w1 = np.zeros((porcent, 1))
w2 = np.zeros((porcent, 1))

print(w1)
print(train_f1)
print(w1*train_f1)
epochs = 1
alpha = 0.0001

while (epochs < 10000):
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    #print(epochs)
    count = 0
    for val in prod:
        if (val >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)

        else:
            cost = 1 - val
            w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1 / epochs * w2)
        count += 1
    epochs += 1

index = list(range(10, 90))
print(index)
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

w1 = w1.reshape(10,1)
w2 = w2.reshape(10,1)


## Extract the test data features
test_f1 = x_test[:, 0]
test_f2 = x_test[:, 1]

test_f1 = test_f1.reshape(10,1)
test_f2 = test_f2.reshape(10,1)

## Predict
y_pred = w1 * test_f1 + w2 * test_f2
print(y_pred)
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print(predictions)
print(accuracy_score(y_test, predictions))