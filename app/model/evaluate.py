from __future__ import division, print_function
import matplotlib.pylab as plt
import numpy as np
from sklearn import svm
import pandas as pd
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.tree import DecisionTreeClassifier


# Combining transductive pu learner and transductive svm

def load_anomaly(path):
    df = pd.read_csv(path)
    labels = df['prey_predator']
    examples = df.drop('prey_predator', 1).drop('Animal', 1).fillna(0)
    examples = MinMaxScaler().fit_transform(examples)
    return np.array(examples), np.array(labels)


# N = 6000

known_labels_ratio = 0.1
X, y = load_anomaly('../data/Animal_Data_prey_predator.csv')
N = X.shape[0]
rp = np.random.permutation(int(N/10))

# data_P = X[y==1][rp[:int(len(rp)*known_labels_ratio)]]
data_P = X[y == 1]
data_N = X[y == 0]
# data_U = np.concatenate((X[y==1][rp[int(len(rp)*known_labels_ratio):]], X[y==0]), axis=0)

print("Amount of positive samples: %d" % (data_P.shape[0]))
print("Amount of negative samples: %d" % (data_N.shape[0]))
plt.figure(figsize=(8, 4.5))
plt.scatter(data_N[:, 0], data_N[:, 1], c='k', marker='.', linewidth=1, s=1, alpha=0.5, label='Negative')
plt.scatter(data_P[:, 0], data_P[:, 1], c='b', marker='o', linewidth=0, s=20, alpha=0.5, label='Positive')
plt.grid()
plt.legend()

# model = DecisionTreeClassifier(max_depth=None, max_features=None,
#                                    criterion='gini', class_weight='balanced')

baggingPU = svm.SVC(kernel='linear', probability=True)

# true_labels = np.zeros(shape=(data_U.shape[0]))
# true_labels[:int(len(rp)*(1.0-known_labels_ratio))] = 1.0


# With different interactions
training_set = []
# training_set = []
training_y = []

# test_set_P = []
# test_set_N = []
# test_set = []
# test_y = []

p_values = []
r_values = []
n = min(data_P.shape[0], data_N.shape[0])
for i in range(8):
    print(str(i))
    # update training dataset
    # randomly pick one positive and negative to training set
    num_unlabled_P = data_P.shape[0] - 1
    num_unlabled_N = data_N.shape[0] - 1

    pick_label_P = random.randint(0, num_unlabled_P)
    if i == 0:
        training_set = [data_P[num_unlabled_P]]
        training_y = [1]
    else:
        training_set = np.append(training_set, [data_P[pick_label_P]], axis=0)
        training_y = np.append(training_y, [1], axis=0)
    data_P = np.delete(data_P, pick_label_P, 0)

    # print(training_set)

    pick_label_N = random.randint(0, num_unlabled_N)
    training_set = np.append(training_set, [data_N[pick_label_N]], axis=0)
    training_y = np.append(training_y, [0], axis=0)
    data_N = np.delete(data_N, pick_label_N, 0)

    test_set = np.append(data_P, data_N, axis=0)
    test_y = np.append(np.ones(data_P.shape[0]), np.zeros(data_N.shape[0]))

    baggingPU.fit(training_set, training_y)
    y_score = baggingPU.predict(test_set)

    # print(np.int_(test_y))
    # print(y_score)
    test_y = np.array(test_y)
    y_score = np.array(y_score)
    precision = f1_score(test_y, y_score, average="macro")
    # print(precision)
    p_values.append(precision)
    # r_values.append(recall)

print('--------------------------------------')
print(p_values)


range1 = lambda start, end: range(start, end+1)

plt.figure(figsize=(12, 4))
plt.plot(range1(1, len(p_values)), p_values, linewidth=2.0)
# plt.plot(range(20), r_values, linewidth=2.0)
# plt.scatter(data_U[:, 0], data_U[:, 1], c='k', marker='.', linewidth=1, s=1, alpha=0.5, label='Unlabeled')
# plt.scatter(data_P[:, 0], data_P[:, 1], c='b', marker='o', linewidth=0, s=20, alpha=0.5, label='Positive')
plt.xlabel('nums of interactions')
plt.ylabel('f1 score')
plt.grid()
plt.legend()
plt.show()