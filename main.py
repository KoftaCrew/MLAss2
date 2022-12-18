import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import sys


# File where you need to keep the logs


class Unbuffered:

    def __init__(self, stream, file):
        self.stream = stream
        self.file_name = file

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        with open(self.file_name, 'a') as f:
            f.write(data)


def decision_tree():
    # Read & separate features from target columns from data.csv (Banknote Authentication Data)
    data = pd.read_csv('./data.csv')
    acc_np = []
    set_tree_sizes = []
    num_trees = 0
    max_tree = -1
    min_tree = 1e9

    # Repeat 5 times
    for split_rate in range(30, 71, 10):
        for rotation in range(1, 6):
            # Work with a copy of the real data
            df_copy = data.copy()
            df_copy = df_copy.sample(frac=1).reset_index(drop=True)
            features = df_copy.drop('class', axis=1)
            target = df_copy['class']

            # Normalization
            normalized_features = (features - features.mean()) / features.std()

            # Experiment with ratio 25% training and 75% test
            x_train = normalized_features[: int(len(normalized_features) * split_rate / 100):]
            y_train = target[: int(len(target) * split_rate / 100):]

            x_test = normalized_features[int(len(normalized_features) * split_rate / 100)::]
            y_test = target[int(len(target) * split_rate / 100)::]

            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)

            predicted = clf.predict(x_test)
            acc = []
            for i in range(len(predicted)):
                acc.append(predicted[i] == y_test.iloc[i])
            acc = np.array(acc)
            print("Rotation Number %s" % rotation)
            print('Accuracy:', acc.mean() * 100)
            print('Ratio:', len(acc), '/', acc.sum())
            print("Train data size: %s" % len(x_train))
            print("Test data size: %s" % len(x_test))
            print("-" * 100)
            acc_np.append([split_rate, acc.mean()])
            num_trees += 1
            set_tree_sizes.append([len(x_train), clf.tree_.node_count])
            max_tree = max(clf.tree_.node_count, max_tree)
            min_tree = min(clf.tree_.node_count, min_tree)
    acc_np = np.array(acc_np)
    set_tree_sizes = np.array(set_tree_sizes)
    print("Accuracy array:", acc_np)
    print("Average tree size: %s" % (set_tree_sizes[:, 1].sum() / num_trees))
    print("Max tree size: %s" % max_tree)
    print("Min tree size: %s" % min_tree)

    for rate in range(30, 71, 10):
        rate_arr = acc_np[acc_np[:, 0] == rate, 1]
        print("-" * 100)
        print("Rate %s" % rate)
        print("Average accuracy: %s" % rate_arr.mean())
        print("Max accuracy: %s" % rate_arr.max())
        print("Min accuracy: %s" % rate_arr.min())
    print("-" * 100)

    plt.scatter(acc_np[:, 1], acc_np[:, 0])
    plt.show()
    plt.scatter(set_tree_sizes[:, 1], set_tree_sizes[:, 0])
    plt.show()

    for i in range(2):
        if i == 0:
            df = pd.DataFrame(acc_np, columns=["Data Split Rate", "Accuracy"])
        else:
            df = pd.DataFrame(set_tree_sizes, columns=["Training Set Size", "Accuracy"])
        f = plt.figure(figsize=(19, 15))
        plt.matshow(df.corr(), fignum=f.number)
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=32,
                   rotation=45)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=32)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=32)
        plt.title('Correlation Matrix', fontsize=32)
        plt.show()


def euclidean_distance(x, y):
    return np.sqrt(np.square(x - y).sum())
    # return np.sqrt(distance)


def normalize(data, cols):
    for col in cols:
        if col != 'class':
            data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data


def knn():
    # Read & separate features from target columns from data.csv (Banknote Authentication Data)
    data = pd.read_csv('./data.csv')

    # divide data into 70% training and 30% test
    train_data = data.sample(frac=0.7)
    test_data = data.drop(train_data.index)

    # normalize features separately
    train_data_norm = normalize(train_data, train_data.columns)  # this normalizes each feature separately
    test_data_norm = normalize(test_data, test_data.columns)


    # Euclidean distance

    correct = {k: 0 for k in range(1, 10)}

    for i in range(len(test_data_norm)):
        distances = []

        for j in range(len(train_data_norm)):
            distances.append([euclidean_distance(test_data_norm.iloc[i, :-1], train_data_norm.iloc[j, :-1]),
                              train_data_norm.iloc[j, -1]])

        distances.sort(key=lambda x: x[0])

        for k in range(1, 10):
            distances_k = distances.copy()
            distances_k = distances_k[:k]

            classes = [x[1] for x in distances_k]

            if classes.count(0) > classes.count(1):

                if test_data_norm.iloc[i, -1] == 0:
                    correct[k] = correct.get(k, 0) + 1

            elif classes.count(0) == classes.count(1):
                if test_data_norm.iloc[i, -1] == train_data_norm.iloc[1, -1]:
                    correct[k] = correct.get(k, 0) + 1
            else:

                if test_data_norm.iloc[i, -1] == 1:
                    correct[k] += 1

    for k in correct:
        print("K value: %s" % k)
        print("Accuracy: %s" % (correct.get(k) / len(test_data_norm) * 100))
        print("-" * 100)
    print("Done")
    pass


def main():
    print("Choose an algorithm")
    print("1. Decision Tree")
    print("2. KNN")
    x = int(input())
    sys.stdout = Unbuffered(sys.stdout, "./logKnn.txt")
    decision_tree() if x == 1 else knn()


if __name__ == '__main__':
    main()
