from collections import Counter
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.model_selection import cross_validate
from math import ceil, sqrt, log
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt

random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number
# Vectorized function for hashing for np efficiency
def w(x):
    return np.int(hash(x)) % 1000
h = np.vectorize(w)

#3.2
class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    def __repr__(self, level = 0):
        if self.max_depth == 0:
            print("\t"*level+str(self.pred))
        else:
            print("\t" * level + str(self.features[self.split_idx])
                  + "<" + str(self.thresh))
            self.left.__repr__(level+1)
            self.right.__repr__(level+1)

    def entropy(self, labels):
        total = len(labels)
        if total == 0:
            return 0
        counts = np.unique(labels, return_counts = True)[1]
        if len(counts) == 1:
            return 0
        else:
            p_zero = counts[0]/total
            p_one = counts[1]/total
            return -(p_zero * log(p_zero, 2) + p_one * log(p_one,2))

    def information_gain(self, feature, labels, threshold):
        H_before = self.entropy(labels)
        left, right = [], []
        for i in range(len(labels)):
            if feature[i] < threshold:
                left.append(labels[i])
            else:
                right.append(labels[i])
        H_after = (len(left)*self.entropy(left) +
                   len(right)*self.entropy(right))/len(labels)
        return H_before - H_after

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y)[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y)[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

#3.3
class RandomForest():
    def __init__(self, max_depth = 3, feature_labels = None, n = 200, m=1):
        self.max_depth = max_depth
        self.feature_labels = feature_labels
        self.n = n
        self.m = m
        self.decision_trees = [
            DecisionTree(max_depth, feature_labels)
            for i in range(self.n)
        ]

    def fit(self, features, labels):
        sample = random.sample(range(features.shape[1]), k = ceil(sqrt(self.m)))
        random_features = features[:, sample]
        self.fit_all(random_features, labels)

    def fit_all(self, features, labels):
        for t in self.decision_trees:
            sample = random.choices(range(len(labels)), k=self.n)
            t.fit(features[sample, :], labels[sample])

    def predict(self, features):
        sample = random.sample(range(features.shape[1]), k = ceil(sqrt(self.m)))
        random_features = features[:, sample]
        preds = self.predict_all(random_features)
        return stats.mode(preds)[0][0,:]

    def predict_all(self, features):
        preds = []
        for tree in self.decision_trees:
            preds.append(tree.predict(features))
        return preds

def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features

def evaluate(clf):
    print("Cross validation:")
    cv_results = cross_validate(clf, X, y, cv=5, return_train_score=True)
    train_results = cv_results['train_score']
    test_results = cv_results['test_score']
    avg_train_accuracy = sum(train_results) / len(train_results)
    avg_test_accuracy = sum(test_results) / len(test_results)

    print('averaged train accuracy:', avg_train_accuracy)
    print('averaged validation accuracy:', avg_test_accuracy)
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)

    return avg_train_accuracy, avg_test_accuracy

if __name__ == "__main__":
    #dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data

        path_train = './datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = './datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    #3.4
    if dataset == "spam":
        X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=7)
        dt = DecisionTree(max_depth=19, feature_labels=features)
        dt.fit(X_train, Y_train)
        pred_train = np.rint(dt.predict(X_train))
        acc_train = sum([x == y for x, y in zip(pred_train, Y_train)])/len(Y_train)
        print("DT Train Acc:" + str(acc_train))
        pred_val = np.rint(dt.predict(X_val))
        acc_val = sum([x == y for x, y in zip(pred_val, Y_val)])/len(Y_val)
        print("DT Val Acc:" + str(acc_val))
        df_spam = pd.DataFrame({"Category": np.rint(dt.predict(Z)).astype(int)})
        df_spam.index = df_spam.index + 1
        df_spam.to_csv("spam" + "submission.csv", index_label="Id")

        rf = RandomForest(max_depth=14, feature_labels=features, m=4, n=400)
        rf.fit(X_train, Y_train)
        pred_train = np.rint(rf.predict(X_train))
        acc_train = sum([x == y for x, y in zip(pred_train, Y_train)]) / len(Y_train)
        print("RF Train Acc:" + str(acc_train))
        pred_val = np.rint(rf.predict(X_val))
        acc_val = sum([x == y for x, y in zip(pred_val, Y_val)]) / len(Y_val)
        print("RF Val Acc:" + str(acc_val))

    if dataset == "titanic":
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=0.2, random_state=7)
        dt = DecisionTree(max_depth=6, feature_labels=features)
        dt.fit(X_train, Y_train)
        pred_train = np.rint(dt.predict(X_train))
        acc_train = sum([x == y for x, y in zip(pred_train, Y_train)]) / len(Y_train)
        print("DT Train Acc:" + str(acc_train))
        pred_val = np.rint(dt.predict(X_val))
        acc_val = sum([x == y for x, y in zip(pred_val, Y_val)]) / len(Y_val)
        print("DT Val Acc:" + str(acc_val))
        df_spam = pd.DataFrame({"Category": np.rint(dt.predict(Z)).astype(int)})
        df_spam.index = df_spam.index+1
        df_spam.to_csv("titanic" + "submission.csv", index_label="Id")

        rf = RandomForest(max_depth=12, feature_labels=features, m=64, n=200)
        rf.fit(X_train, Y_train)
        pred_train = np.rint(rf.predict(X_train))
        acc_train = sum([x == y for x, y in zip(pred_train, Y_train)]) / len(Y_train)
        print("RF Train Acc:" + str(acc_train))
        pred_val = np.rint(rf.predict(X_val))
        acc_val = sum([x == y for x, y in zip(pred_val, Y_val)]) / len(Y_val)
        print("RF Val Acc:" + str(acc_val))

    #3.5.3
    if dataset == "spam":
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=0.2, random_state=7)
        accs = []
        for i in range(1, 40):
            dt = DecisionTree(max_depth = i, feature_labels=features)
            dt.fit(X_train, Y_train)
            pred = np.rint(dt.predict(X_val))
            accs.append(sum([x == y for x, y in zip(pred, Y_val)])/len(Y_val))
        plt.plot(np.arange(1, 40), accs)
        plt.xlabel("Max Depth")
        plt.title("Max Depth vs Accuracy for Spam")
        plt.ylabel("Accuracy")
        plt.show()

    #3.6
    if dataset == "titanic":
        print("\n Tree Structure")
        t = DecisionTree(max_depth = 3, feature_labels = features)
        t.fit(X,y)
        t.__repr__()