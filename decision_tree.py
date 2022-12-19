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


def experiment1():
    # Read & separate features from target columns from data.csv (Banknote Authentication Data)
    data = pd.read_csv('./data.csv')
    output = pd.DataFrame([], columns=["Rotation", "Split rate", "Accuracy", "Tree size"])

    split_rate = 25
    # Repeat 5 times
    for rotation in range(1, 6):
        # Work with a copy of the real data
        df_copy = data.copy()
        df_copy = df_copy.sample(frac=1).reset_index(drop=True)
        features = df_copy.drop('class', axis=1)
        target = df_copy['class']

        # Experiment with ratio 25% training and 75% test
        x_train = features[: int(len(features) * split_rate / 100):]
        y_train = target[: int(len(target) * split_rate / 100):]

        x_test = features[int(len(features) * split_rate / 100)::]
        y_test = target[int(len(target) * split_rate / 100)::]

        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        predicted = clf.predict(x_test)
        acc = []
        for i in range(len(predicted)):
            acc.append(predicted[i] == y_test.iloc[i])
        acc = np.array(acc)
        print(f"Split Rate: {split_rate}% train {100 - split_rate}% test")
        print(f"Rotation Number {rotation}")
        print(f"Accuracy: {acc.mean() * 100}%")
        print(f"Tree Size: {clf.tree_.node_count}")
        print("-" * 100)

        output = pd.concat([output, pd.DataFrame({
                "Rotation": [rotation],
                "Split rate": [split_rate],
                "Accuracy": [acc.mean() * 100],
                "Tree size": [clf.tree_.node_count]
            })])

    output.to_csv("decision_experiment1.csv", index=False)
    print(output.head())


def experiment2():
    # Read & separate features from target columns from data.csv (Banknote Authentication Data)
    data = pd.read_csv('./data.csv')
    acc_np = []
    set_tree_sizes = []
    num_trees = 0
    max_tree = -1
    min_tree = 1e9

    report = pd.DataFrame([], columns=["Rotation", "Split", "Accuracy"])
    # Repeat 5 times
    for split_rate in range(30, 71, 10):
        for rotation in range(1, 6):
            # Work with a copy of the real data
            df_copy = data.copy()
            df_copy = df_copy.sample(frac=1).reset_index(drop=True)
            features = df_copy.drop('class', axis=1)
            target = df_copy['class']

            # Experiment with ratio 25% training and 75% test
            x_train = features[: int(len(features) * split_rate / 100):]
            y_train = target[: int(len(target) * split_rate / 100):]

            x_test = features[int(len(features) * split_rate / 100)::]
            y_test = target[int(len(target) * split_rate / 100)::]

            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)

            predicted = clf.predict(x_test)
            acc = []
            for i in range(len(predicted)):
                acc.append(predicted[i] == y_test.iloc[i])
            acc = np.array(acc)
            print(f"Split Rate: {split_rate}% train {100 - split_rate}% test")
            print(f"Rotation Number {rotation}")
            print(f"Accuracy: {acc.mean() * 100}%")

            report = pd.concat([
                report,
                pd.DataFrame({
                    "Rotation": [rotation],
                    "Split": [split_rate],
                    "Accuracy": [acc.mean() * 100]
                })
            ])

            print("-" * 100)
            acc_np.append([split_rate, acc.mean()])
            num_trees += 1
            set_tree_sizes.append([len(x_train), clf.tree_.node_count])
            max_tree = max(clf.tree_.node_count, max_tree)
            min_tree = min(clf.tree_.node_count, min_tree)

    report.to_csv("decision_experiment2_accuracy.csv", index=False)
    acc_np = np.array(acc_np)
    set_tree_sizes = np.array(set_tree_sizes)
    print("Accuracy array:", acc_np)
    print("Average tree size: %s" % (set_tree_sizes[:, 1].sum() / num_trees))
    print("Max tree size: %s" % max_tree)
    print("Min tree size: %s" % min_tree)

    report = pd.DataFrame([], columns=["Split", "Average accuracy", "Max accuracy", "Min accuracy"])
    for rate in range(30, 71, 10):
        rate_arr = acc_np[acc_np[:, 0] == rate, 1]
        print("-" * 100)
        print("Rate %s" % rate)
        print("Average accuracy: %s" % rate_arr.mean())
        print("Max accuracy: %s" % rate_arr.max())
        print("Min accuracy: %s" % rate_arr.min())
        report = pd.concat([
            report,
            pd.DataFrame({
                "Split": [rate],
                "Average accuracy": [rate_arr.mean()],
                "Max accuracy": [rate_arr.max()],
                "Min accuracy": [rate_arr.min()]
            })
        ])
    report.to_csv("decision_experiment2_accuracies_across_rates.csv", index=False)
    print("-" * 100)

    plt.scatter(acc_np[:, 1], acc_np[:, 0])
    plt.title("Training set split percentage against accuracy")
    plt.ylabel("Split")
    plt.xlabel("Accuracy")
    plt.show()

    plt.scatter(set_tree_sizes[:, 1], set_tree_sizes[:, 0])
    plt.title("Nodes against training set size")
    plt.ylabel("Train size")
    plt.xlabel("Node count")
    plt.show()

    df = data.copy()
    f = plt.figure(figsize=(10, 10))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=19,
               rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=19)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=19)
    plt.title('Correlation Matrix', fontsize=19)
    plt.show()


if __name__ == '__main__':
    sys.stdout = Unbuffered(sys.stdout, "./logDecisionTree.txt")
    print("-" * 100)
    print("Experiment 1:\n")
    experiment1()
    print()
    print("-" * 100)
    print("Experiment 2:\n")
    experiment2()
