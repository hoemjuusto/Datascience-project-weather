import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import collections
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error




def knn_optimize(self, show_plot=False):
    """
    Finds the optimal minimum number of neighbors to use for the KNN classifier.
    :param show_plot: bool, when True shows the plot of number of neighbors vs error
            Default: False
    :return: the number of neighbors (int)
    """
    df = self
    X = df.drop(['name', 'labels'], axis=1)
    y = df['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=seed1)

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    m = min(error)
    min_ind = error.index(m)

    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()

    return min_ind + 1


if __name__ == "__main__":

    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    Y_train = pd.read_csv('../res/weather_data_train_labels.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)

    X_test = pd.read_csv('../res/weather_data_test.csv', index_col='datetime',
                         sep=';', decimal=',', infer_datetime_format=True)

    Y_test = pd.read_csv('../res/weather_data_test_labels.csv', index_col='datetime',
                         sep=';', decimal=',', infer_datetime_format=True)

    # lets standardize our data first
    X_std = StandardScaler().fit_transform(X_train)
    # Y_std = StandardScaler().fit_transform(Y_train)

    X_test_std = StandardScaler().fit_transform(X_test)
    # Y_test_std = StandardScaler().fit_transform(Y_test)

    pca_func = PCA(n_components=8)
    PCA_cmps = pca_func.fit_transform(X_std)
    test_PCA_cmps = pca_func.fit_transform(X_test_std)

    x_train, x_test, y_train, y_test = PCA_cmps, test_PCA_cmps, Y_train['OBSERVED'], Y_test['OBSERVED']

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))

    m = min(error)
    min_ind = error.index(m)
    print("Value of K that produces minimal error: ", min_ind + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    # We find that k=31 produces minimal error
    knn = KNeighborsClassifier(n_neighbors=31)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)

    final_df = pd.DataFrame(data={x_train[:, 0], x_train[:, 1], y_train})
    pred_df = pd.DataFrame(data={x_test[:, 0], x_test[:, 1], pred_i, y_test})

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=15)

    for i, row in final_df.iterrows():
        if row['OBSERVED'] == 1:
            ax.scatter(row[0], row[1], c='r', s=5)
        else:
            ax.scatter(row[0], row[1], c='b', s=5)

    for i, row in pred_df.iterrows():
        if row['OBSERVED'] != row[2]:
            ax.scatter(row[0], row[1], 'x', c='black', s=10)
        else:
            if row[2] == 1:
                ax.scatter(row[0], row[1], c='y', s=5)
            else:
                ax.scatter(row[0], row[1], c='g', s=5)

    plt.show()