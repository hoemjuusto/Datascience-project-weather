import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.pca import pca
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



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

    mean_error = np.mean((regression_frame.loc[:, 'predicted U_mu'] - regression_frame.loc[:, 'U_mu'])
                         / regression_frame.loc[:, 'U_mu'])

    print("Mean error in predicted humidity before PCA: ", mean_error)
    print("Error count with predicted OBSERVED: ", sum(np.abs(regression_frame.iloc[:, 1]
                                                              - regression_frame.iloc[:, 0])),
          " out of total 3140 observation\n")

    # linear regression with n PCA-components
    n = 4
    pca_model = LinearRegression()
    pca_cmps, std_df = pca(X_train, n_comp=n)
    pca_model.fit(pca_cmps, Y_train)
    test_pca_set, std_test_df = pca(X_test, n_comp=n)
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
                     - pca_regression_frame.iloc[:, 0])), " out of total 3140 observation\n")
