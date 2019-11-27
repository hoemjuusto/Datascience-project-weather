from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                              sep=';', decimal=',', infer_datetime_format=True)
    Y_train = pd.read_csv('../res/weather_data_train_labels.csv', index_col='datetime',
                              sep=';', decimal=',', infer_datetime_format=True)

    X_test = pd.read_csv('../res/weather_data_test.csv', index_col='datetime',
                              sep=';', decimal=',', infer_datetime_format=True)
    Y_test = pd.read_csv('../res/weather_data_test_labels.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)

    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    prediction_df = pd.DataFrame(data={'predicted OBS': np.rint(predictions[:, 0]),
                                                            'OBSERVED': Y_test['OBSERVED'],
                                                            'predicted U_mu': predictions[:, 1],
                                                            'U_mu': Y_test['U_mu']})
    print(prediction_df)
