from itertools import combinations
from tqdm import tqdm
from PyCausality.TransferEntropy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kde(df, endog, exog):
    """
    Calculate TE using KDE
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=2)
    te.nonlinear_TE(pdf_estimator='kernel', n_shuffles=20)
    result = te.results.transpose().tolist()
    #result = te.results.values.reshape((4, 2)).transpose().tolist()
    result[0] = [X, Y] + result[0]
    result[1] = [Y, X] + result[1]
    return result


def granger(df, endog, exog):
    """
    Calculate TE using granger causality
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=2)
    te.linear_TE(n_shuffles=20)
    result = te.results.transpose().tolist()
    result[0] = [X, Y] + result[0]
    result[1] = [Y, X] + result[1]
    return result


def mi(df, endog, exog):
    """
    Calculate Mutual Information using KDE
    Creates calculation results of MI into a list as [X, Y, MI_XY, p_value_XY, z_score_XY, Ave_MI_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=2, method='mi')
    te.mutual_information(pdf_estimator='kernel', n_shuffles=20)
    in_result = te.results.transpose().tolist()
    result = [[X, Y] + in_result[0]]
    return result


def cluster_relation_cal(method, period):
    stock_data = pd.read_csv('Data/hierarch_mean_df/{}_{}_cluster_mean.csv'.format(method, period))
    stock_data.drop('Unnamed: 0', axis=1, inplace=True)
    comb_list = [combs for combs in combinations(stock_data.columns, 2)]
    print('total combinations are: ' + str(len(comb_list)))
    calculation = []
    for comb_no in tqdm(range(len(comb_list)), unit='combination', desc='calculating'):
        X = comb_list[comb_no][0]
        Y = comb_list[comb_no][1]
        reduced_data = stock_data[[Y, X]].dropna().values
        if method == 'kernel':
            result_list = kde(reduced_data, Y, X)
        elif method == 'mi':
            result_list = mi(reduced_data, Y, X)
        else:
            result_list = granger(reduced_data, Y, X)
        calculation = calculation + result_list  # add the lists into a larger list
    if method == 'mi':
        result_df = pd.DataFrame(calculation, columns=['exog', 'endog', 'MI_XY', 'Norm_MI_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_MI_XY'])
    else:
        result_df = pd.DataFrame(calculation, columns=['exog', 'endog', 'TE_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_TE_XY'])
        result_df['ETE_XY'] = result_df['TE_XY'] - result_df['Ave_TE_XY']
    return result_df

methods = ['mi', 'correlation', 'kernel', 'granger']
periods = ['2017_1', '2017_2', '2018_1', '2018_2', '2019_1', '2019_2', '2020_1', '2020_2', '2021_1', '2021_2']
#periods = ['2018_1', '2018_2', '2019_1', '2019_2', '2020_1', '2020_2']
comp_mean_mi = []
#for period in periods:
for method in methods:
    print('%' * 60)
    print(' ' * 25 + method + ' ' * 25)
    print('%' * 60)
    pd_mean_mi = []
    for period in periods:
    #for method in methods:
        print('=' * 20 + ' ' * 5 + period + ' ' * 5 + '=' * 20)
        """
        cluster-wise realtionship calculation
        hierarch/louvain
        ker/mi/granger
        """
        #result_df = cluster_relation_cal(method, period)
        #result_df.to_csv('Data/hierarch_cluster_ker/{}_{}_inter_causality.csv'.format(method, period))

        mi_df = pd.read_csv('Data/louvain_cluster_ker/{}_{}_inter_causality.csv'.format(method, period))
        mi_df.drop('Unnamed: 0', axis=1, inplace=True)
        mi_df = mi_df[mi_df['p_value_XY'] <= 0.05]
        #mi_df[mi_df['ETE_XY'] < 0] = 0

        max_num = max(mi_df['endog'].values) + 1
        mi_matrix = np.empty((max_num, max_num))
        """
        for i in range(max_num):
            for j in range(max_num):
                if i < j:
                    mi_matrix[i][j] = mi_df.loc[mi_df['exog'] == i].loc[mi_df['endog'] == j]['Norm_MI_XY'].values[0]
                elif i == j:
                    mi_matrix[i][j] = 1.0
                else:
                    mi_matrix[i][j] = mi_matrix[j][i]
        """
        for i in range(max_num):
            for j in range(max_num):
                if i == j:
                    mi_matrix[i][j] = 1.0
                else:
                    mi_matrix[i][j] = mi_df.loc[mi_df['exog'] == i].loc[mi_df['endog'] == j]['ETE_XY'].values[0]

        plt.matshow(mi_matrix)
        plt.title('{}_{}'.format(method, period))
        plt.colorbar(shrink=0.8)
        plt.clim(0.0, 0.4)
        for (i, j), z in np.ndenumerate(mi_matrix):
            plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.show()
        """
        mean_mi = mi_df['Norm_MI_XY'].mean()
        pd_mean_mi.append(mean_mi)
    comp_mean_mi.append(pd_mean_mi)
    plt.plot(pd.DataFrame(comp_mean_mi, index=methods).T)
    plt.legend(methods)
    plt.show()
    """




