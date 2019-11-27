import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pca(df):

    std_df = df
    # standardize
    for col in std_df:
        mean = np.mean(std_df[col])
        std = np.std(std_df[col])
        for i in range(len(std_df[col])):
            std_df[col][i] = (std_df[col][i] - mean) / std
    # normalize
    for col in std_df:
        max = np.max(std_df[col])
        min = np.min(std_df[col])
        for i in range(len(std_df[col])):
            std_df[col][i] = (std_df[col][i] - min) / (max - min)

    # covariance matrix out of standardized data matrix
    cov_matrix = std_df.T @ std_df
    # eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    # sorting eigenVectors and -values according to descending order by eigenvalue
    idx = eigenValues.argsort()[::-1]
    sorted_eigenValues = eigenValues[idx]
    sorted_eigenVectors = eigenVectors[:, idx]
    # new dataframe for pca
    pca_df = pd.DataFrame(index=df.index)
    for i in range(len(sorted_eigenVectors)):
        pca_df['PCA_cmp' + str(i + 1)] = std_df.to_numpy() @ sorted_eigenVectors[i]

    return pca_df


if __name__ == "__main__":

    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    test = X_train.head(500)
    test_pca = pca(test)
    print(test_pca)
    first2 = test_pca.loc[:, ('PCA_cmp1', 'PCA_cmp2')]
    plt.scatter(first2['PCA_cmp1'], first2['PCA_cmp2'], s=1)
    plt.show()
