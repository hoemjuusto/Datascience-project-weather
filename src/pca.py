import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(df, n_comp=0, expl_var=False):
    # standardize
    std_df = StandardScaler().fit_transform(df)
    # covariance matrix out of standardized data matrix
    N = len(df)
    cov_matrix = 1/(N-1)*std_df.T @ std_df
    # eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    # sorting eigenVectors and -values according to descending order by eigenvalue
    idx = eigenValues.argsort()[::-1]
    sorted_eigenValues = eigenValues[idx]
    sorted_eigenVectors = -eigenVectors[:, idx]

    if(expl_var):
        """Variance explained by PCA-components (= How much of the difference between data samples is still captured when
        projecting onto eigenvector defined lines"""
        tot = sum(eigenValues)
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
        pca_df['PCA_cmp' + str(i + 1)] = sorted_eigenVectors[:, i] @ std_df.T

    if n_comp <= 0:
        return pca_df, sorted_eigenVectors
    else:
        return pca_df.iloc[:, 0:n_comp], sorted_eigenVectors


if __name__ == "__main__":

    X_train = pd.read_csv('../res/weather_data_train.csv', index_col='datetime',
                          sep=';', decimal=',', infer_datetime_format=True)
    test = X_train
    # testing self made pca function
    test_pca, stand_df = pca(test)
    print(test_pca)
    first2 = test_pca.loc[:, ('PCA_cmp1', 'PCA_cmp2')]
    plt.scatter(first2.loc[:, 'PCA_cmp1'], first2.loc[:, 'PCA_cmp2'], s=1)
    plt.show()


    # testing default pca function from sklearn
    def_pca = PCA()
    X_std = StandardScaler().fit_transform(X_train)
    stdf = pd.DataFrame(data=X_std)
    pcaComps = def_pca.fit_transform(X_std)
    pcdf = pd.DataFrame(data=pcaComps)
    print(pcdf)
    def_first2 = pcdf.loc[:, 0:1]
    print(def_first2)
    plt.scatter(def_first2.loc[:, 0], def_first2.loc[:, 1], s=1)
    plt.show()

