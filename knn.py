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

    def flush(self):
        self.stream.flush()


def euclidean_distance(x, y):
    return np.sqrt(np.square(x - y).sum())


def normalize(data: pd.DataFrame, mean: pd.Series, std: pd.Series):
    return (data - mean) / std


def main():
    # Read & separate features from target columns from data.csv
    data = pd.read_csv('./data.csv')

    # divide data into 70% training and 30% test
    train_data = data.sample(frac=0.7)
    x_train_data = train_data.drop('class', axis=1)
    y_train_data = train_data['class']
    test_data = data.drop(train_data.index)
    x_test_data = test_data.drop('class', axis=1)
    y_test_data = test_data['class']

    # normalize features separately
    mean = x_train_data.mean()
    std = x_train_data.std()
    train_data_norm = normalize(x_train_data, mean, std)
    test_data_norm = normalize(x_test_data, mean, std)


    # Euclidean distance

    correctList = [0] * 10

    for i in range(len(test_data_norm)):
        distances = []

        for j in range(len(train_data_norm)):
            distances.append([euclidean_distance(test_data_norm.iloc[i], train_data_norm.iloc[j]),
                              train_data_norm.index[j], y_train_data.iloc[j]])

        distances.sort(key=lambda x: x[1])
        distances.sort(key=lambda x: x[0])

        for k in range(1, 10):
            distances_k = distances.copy()
            distances_k = distances_k[:k]

            classes = [x[2] for x in distances_k]

            if classes.count(0) > classes.count(1):
                if y_test_data.iloc[i] == 0:
                    correctList[k] += 1
            elif classes.count(0) == classes.count(1):
                if y_test_data.iloc[i] == classes[0]:
                    correctList[k] += 1
            else:
                if y_test_data.iloc[i] == 1:
                    correctList[k] += 1

    for k, correct in enumerate(correctList[1:]):
        print(f"k value: {k + 1}")
        print(f"Number of correctly classified instances: {correct} Total number of instances: {len(test_data_norm)}")
        print(f"Accuracy: {correct / len(test_data_norm)}")
        print()


if __name__ == '__main__':
    sys.stdout = Unbuffered(sys.stdout, "./logKnn.txt")
    main()
