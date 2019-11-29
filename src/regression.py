import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.pca import pca
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":

    model = LinearRegression()
    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    Y_train = pd.read_csv('../res/weather_data_train_labels.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)

    X_test = pd.read_csv('../res/weather_data_test.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)

    Y_test = pd.read_csv('../res/weather_data_test_labels.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    # linear regression
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    predictions[:, 0] = np.rint(predictions[:, 0])
    regression_frame = pd.DataFrame(index=X_test.index, data={'predicted OBSERVATION': predictions[:, 0],
                                                              'OBSERVATION': Y_test.loc[:, 'OBSERVED'],
                                                              'predicted U_mu': predictions[:, 1],
                                                              'U_mu': Y_test.loc[:, 'U_mu']})

    print(regression_frame.head(10))

    mse = mean_squared_error(Y_test.loc[:, 'U_mu'], predictions[:, 1])

    print("Mean error in predicted humidity before PCA: ", mse)
    print("Error count with predicted OBSERVED: ", sum(np.abs(regression_frame.iloc[:, 1]
                                                              - regression_frame.iloc[:, 0])),
          " out of total 3140 observation\n")

    # linear regression with n PCA-components (own pca-function)
    """n = 16
    pca_model = LinearRegression()
    pca_cmps, std_df = pca(X_train)
    pca_model.fit(pca_cmps, Y_train)
    test_pca_set, std_test_df = pca(X_test)
    pca_predictions = pca_model.predict(test_pca_set)
    pca_predictions[:, 0] = np.rint(pca_predictions[:, 0])
    pca_regression_frame = pd.DataFrame(index=X_test.index, data={'predicted OBSERVATION': pca_predictions[:, 0],
                                                                  'OBSERVATION': Y_test.loc[:, 'OBSERVED'],
                                                                  'predicted U_mu': pca_predictions[:, 1],
                                                                  'U_mu': Y_test.loc[:, 'U_mu']})
    print(pca_regression_frame.head(10))

    mean_error = np.mean((pca_regression_frame.loc[:, 'predicted U_mu'] - pca_regression_frame.loc[:, 'U_mu'])
                         / pca_regression_frame.loc[:, 'U_mu'])

    print("Mean error in humidity with ", n, " first PCA-components: ", mean_error)
    print("Error count with predicted OBSERVED: ",
          sum(np.abs(pca_regression_frame.iloc[:, 1]
                     - pca_regression_frame.iloc[:, 0])), " out of total 3140 observation\n")"""


    # linear regression with build in PCA
    # lets standardize our data first
    X_std = StandardScaler().fit_transform(X_train)
    # Y_std = StandardScaler().fit_transform(Y_train)

    X_test_std = StandardScaler().fit_transform(X_test)
    # Y_test_std = StandardScaler().fit_transform(Y_test)

    pca_func = PCA()
    PCA_cmps = pca_func.fit_transform(X_std)
    cmp_index = ["PCA_cmp"+str(i) for i in range(1, 17)]
    pca_df = pd.DataFrame(index=X_train.index, data=PCA_cmps)

    expl_var = pca_func.explained_variance_ratio_
    cum_expl_var = np.cumsum(pca_func.explained_variance_ratio_)
    variance_df = pd.DataFrame(index=cmp_index, data={'explained variance': expl_var,
                                                      'cumulative exp var': cum_expl_var})

    print(variance_df)

    plt.plot(range(1, 17), cum_expl_var)
    # We see that keeping only 8 components explains over 90% of the variance
    pca_df = pd.DataFrame(index=X_train.index, data=pca_df.iloc[:, 0:8])
    print("pca df\n", pca_df)
    final_df = pd.concat([pca_df, Y_train['OBSERVED']], axis=1)
    print(final_df)

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

    plt.show()

    # now lets do linear regression with the 8 PCA comps
    pca_func = PCA(n_components=8)
    PCA_cmps = pca_func.fit_transform(X_std)

    pca_model = LinearRegression()
    pca_model.fit(PCA_cmps, Y_train)

    # we have to do the PCA also for the test data
    test_PCA_cmps = pca_func.fit_transform(X_test_std)
    predictions = pca_model.predict(test_PCA_cmps)

    pca_regression_frame = pd.DataFrame(index=X_test.index, data={'predicted OBSERVATION': predictions[:, 0],
                                                                  'OBSERVATION': Y_test.loc[:, 'OBSERVED'],
                                                                  'predicted U_mu': predictions[:, 1],
                                                                  'U_mu': Y_test.loc[:, 'U_mu']})
    print(pca_regression_frame.head(10))

    mse = mean_squared_error(Y_test.loc[:, 'U_mu'], predictions[:, 1])

    print("Mean squared error in humidity with ", str(8), " first PCA-components: ", mse)
    print("Error count with predicted OBSERVED: ",
          sum(np.abs(pca_regression_frame.iloc[:, 1]
                     - pca_regression_frame.iloc[:, 0])), " out of total 3140 observation\n")














