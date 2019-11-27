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
    """ Variance covered by pca_components (How much of the variability between data samples is still present, after
    projecting onto lines defined by eigenvectors"""
    tot = sum(sorted_eigenValues)
    print("\n", tot)
    var_exp = [(i / tot) * 100 for i in sorted_eigenValues]
    print("\n\n1. Variance Explained\n", var_exp)
    cum_var_exp = np.cumsum(var_exp)
    print("\n\n2. Cumulative Variance Explained\n", cum_var_exp)
    print("\n\n3. Percentage of variance the first two principal components each contain\n ", var_exp[0:2])
    print("\n\n4. Percentage of variance the first two principal components together contain\n", sum(var_exp[0:2]))
    # new dataframe for pca
    pca_df = pd.DataFrame(index=df.index)
    for i in range(len(sorted_eigenVectors)):
        pca_df['PCA_cmp' + str(i + 1)] = std_df @ sorted_eigenVectors[:, i]

    return pca_df, std_df


if __name__ == "__main__":

    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    test = X_train
    test_pca, std_df = pca(test)
    print(test_pca)
    first2 = test_pca.loc[:, ('PCA_cmp1', 'PCA_cmp2')]
    plt.scatter(first2['PCA_cmp1'], first2['PCA_cmp2'], s=1)
    plt.show()
