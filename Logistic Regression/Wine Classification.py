import scipy.io
from sklearn.model_selection import train_test_split
from scipy.special import expit
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Load,normalize, and split
wine = scipy.io.loadmat('data/data.mat')
wine_X = wine["X"]
X_test = wine["X_test"]
wine_Y = wine["y"]
X_train, X_val, Y_train, Y_val = train_test_split(wine_X, wine_Y, random_state = 5)
Y_val = Y_val.flatten()

def n(x):
    a = [np.std(x[:, y]) for y in np.arange(len(x[0]))]
    b = [np.mean(x[:,y]) for y in np.arange(len(x[0]))]
    for c in range(len(x[0])):
        x[:,c] = x[:,c] - b[c]
        x[:,c] = np.true_divide(x[:,c], a[c])
n(X_train)
n(X_val)
n(X_test)

X_train = np.append(X_train, np.ones((len(X_train),1)), axis = 1)
X_val = np.append(X_val, np.ones((len(X_val),1)), axis = 1)
X_test = np.append(X_test, np.ones((len(X_test),1)), axis = 1)

def predict(X, w):
    pred = []
    for i in range(len(X)):
        pred.append(scipy.special.expit(np.dot(X[i], w)))
    return np.array(pred)

#2 Batch Gradient Descent
def main_q2():
    weights = np.zeros((len(X_train[0]),))
    losses = []
    learn_rate = .001
    reg = 0.1
    pred = predict(X_train, weights)
    losses.append(-np.dot(Y_train.flatten(), np.log(pred + 1e-12)) - np.dot(
        ((1-Y_train.flatten()) + 1e-12), np.log((1 - pred) + 1e-12)) +
                (reg/2)*np.sum(np.square(weights)))
    for i in range(4000):
        weights = weights - learn_rate * (reg * weights
                                          - np.dot(np.transpose(X_train),
                                                   Y_train.flatten()-pred))
        pred = predict(X_train, weights)
        losses.append(-np.dot(Y_train.flatten(), np.log(pred + 1e-12)) - np.dot(
            ((1 - Y_train.flatten()) + 1e-12), np.log((1 - pred) + 1e-12)) +
                      (reg / 2) * np.sum(np.square(weights)))

    plt.plot(np.arange(4000+1), losses)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Iterations vs Loss of BGD")
    plt.show()

    acc = []
    pred_val = np.rint(predict(X_val, weights))
    for i in pred_val:
        for j in Y_val:
            acc.append(i == j)
    print("Val acc: " + str(sum(acc) / len(Y_val)))
    return weights

#4 Stochastic Gradient Descent
def main_q4():
    weights = np.zeros((len(X_train[0]),))
    losses = []
    learn_rate = .000001
    reg = 0.1
    pred = predict(X_train, weights)
    losses.append(-np.dot(Y_train.flatten(), np.log(pred + 1e-12)) - np.dot(
        ((1-Y_train.flatten()) + 1e-12), np.log((1 - pred) + 1e-12)) +
                (reg/2)*np.sum(np.square(weights)))
    np.random.seed(5)
    index = -1
    np.random.shuffle(X_train)
    for i in range(5000):
        index = index + 1
        if len(X_train) == index:
            index = 0
            np.random.shuffle(X_train)
        weights = weights - learn_rate * (reg * weights - len(X_train)*
                                           (Y_train.flatten()[index]-pred[index])*
                                           np.transpose(X_train[index]))
        pred = predict(X_train, weights)
        losses.append(-np.dot(Y_train.flatten(), np.log(pred + 1e-12)) - np.dot(
            ((1 - Y_train.flatten()) + 1e-12), np.log((1 - pred) + 1e-12)) +
                      (reg / 2) * np.sum(np.square(weights)))

    plt.plot(np.arange(5000+1), losses)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Iterations vs Loss of SGD")
    plt.show()

def main_q5():
    weights = np.zeros((len(X_train[0]),))
    losses = []
    learn_rate = .001
    reg = 0.1
    pred = predict(X_train, weights)
    losses.append(-np.dot(Y_train.flatten(), np.log(pred + 1e-12)) - np.dot(
        ((1-Y_train.flatten()) + 1e-12), np.log((1 - pred) + 1e-12)) +
                (reg/2)*np.sum(np.square(weights)))
    np.random.seed(6)
    index = -1
    np.random.shuffle(X_train)
    for i in range(5000):
        index = index + 1
        if len(X_train) == index:
            index = 0
            np.random.shuffle(X_train)
        weights = weights - learn_rate/(i+1) * (reg * weights - len(X_train)*
                                           (Y_train.flatten()[index]-pred[index])*
                                           np.transpose(X_train[index]))
        pred = predict(X_train, weights)
        losses.append(-np.dot(Y_train.flatten(), np.log(pred + 1e-12)) - np.dot(
            ((1 - Y_train.flatten()) + 1e-12), np.log((1 - pred) + 1e-12)) +
                      (reg / 2) * np.sum(np.square(weights)))

    plt.plot(np.arange(5000+1), losses)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Iterations vs Loss of SGD with Decay")
    plt.show()

def main_q6():
    df_wine = pd.DataFrame({"Category": np.rint(predict(X_test, main_q2())).astype(int)})
    df_wine.index = df_wine.index + 1
    df_wine.to_csv("wine" + "submission.csv", index_label="Id")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=int, default=1, help="Specify which question to run")
    args = parser.parse_args()
    if args.question == 2:
        main_q2()
    elif args.question == 4:
        main_q4()
    elif args.question == 5:
        main_q5()
    elif args.question == 6:
        main_q6()
    else:
        raise ValueError("Cannot find specified question number")
