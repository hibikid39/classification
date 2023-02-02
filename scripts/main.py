import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def train(filepath):
    df = pd.read_csv(filepath, index_col=0)

    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    target_names = ["sample0", "sample1", "sample2"]
    #print(X)
    #print(y)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    # PCA
    pca = PCA(n_components=3)
    pca = pca.fit(X)
    X_r = pca.transform(X)
    print("explained variance ratio (first three components): {}".format(pca.explained_variance_ratio_))

    # plot
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    plt.savefig("pca_train.png")

    knn = KNeighborsClassifier(n_neighbors=3) # num of class = 3
    knn.fit(X_r, y)

    return pca, knn

def test(filepath, pca, knn):
    df = pd.read_csv(filepath, index_col=0)

    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    target_names = ["sample0", "sample1", "sample2"]
    #print(X)
    #print(y)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    # PCA
    X_r = pca.transform(X)

    # plot
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    plt.savefig("pca_test_gt.png")

    y_pred = knn.predict(X_r)

    return y_pred

def main():
    pca, knn = train("data/sample_train.csv")
    y_pred = test("data/sample_test.csv", pca, knn)

    print(y_pred)

if __name__ == "__main__":
    main()
