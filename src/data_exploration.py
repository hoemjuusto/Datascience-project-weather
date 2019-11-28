import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# following to get the dates appropriate size on x-axis
import matplotlib
matplotlib.rc('xtick', labelsize=7)


if __name__ == "__main__":

    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    Y_train = pd.read_csv('../res/weather_data_train_labels.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)

    combined = pd.concat([X_train, Y_train], axis=1)

    print(X_train.shape)
    print(X_train.head(10))
    print(Y_train.shape)
    print(Y_train.head(10))
    print(combined.shape)
    print(combined.head(10))

    # X_train = X_train.head(50)    # to see results with fewer data points

    index_col = X_train.index   # index column as x-axis
    Tn_mu = X_train['Tn_mu']    # mean minimum temperature
    Tx_mu = X_train['Tx_mu']    # mean maximum temperature
    # Histograms
    """X_train.hist(column=['Tn_mu', 'Tx_mu'])
    plt.show()"""
    # pairplots
    """sns.pairplot(combined)
    plt.show()"""
    """df = combined   # dataframe to be used to build correlation matrix
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);"""



