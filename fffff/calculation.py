from PyCausality.TransferEntropy import *
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from scipy import stats


def kde(df, endog, exog):
    """
    Calculate TE using KDE
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=1)
    te.nonlinear_TE(pdf_estimator='kernel', n_shuffles=20)
    result = te.results.transpose().tolist()
    #result = te.results.values.reshape((4, 2)).transpose().tolist()
    result[0] = [X, Y] + result[0]
    result[1] = [Y, X] + result[1]
    return result


def histogram(df, endog, exog):
    """
    Calculate TE using Histogram
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=2)
    te.nonlinear_TE(pdf_estimator='histogram', n_shuffles=200)
    result = te.results.values.reshape((4, 2)).transpose().tolist()
    result[0] = [X, Y] + result[0]
    result[1] = [Y, X] + result[1]
    return result


def granger(df, endog, exog):
    """
    Calculate TE using granger causality
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=1)
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
    te = TransferEntropy(df=df, endog=endog, exog=exog, lag=1, method='mi')
    te.mutual_information(pdf_estimator='kernel', n_shuffles=20)
    in_result = te.results.transpose().tolist()
    result = [[X, Y] + in_result[0]]
    return result

def corr(df, endog, exog):
    in_result = stats.pearsonr(df[:, 0], df[:, 1])
    result = [[X, Y] + list(in_result)]
    return result


#period = ['2020_2', '2020_1', '2019_1', '2019_2', '2018_1', '2018_2', '2017_1', '2017_2']
period = ['2021_2', '2021_1', '2020_2', '2020_1', '2019_1', '2019_2', '2018_1', '2018_2', '2017_1', '2017_2']
for files in period:
    print('%' * 60)
    print(' ' * 25 + files + ' ' * 25)
    print('%' * 60)
    # Read stock_return data and change index to datetime index
    stock_data = pd.read_csv('Data/krx350_prior_half_returns/'+files+'_kspkdq_returns.csv').dropna(axis='columns', how='all')
    stock_data.drop('Unnamed: 0', axis=1, inplace=True)
    # drop the stocks that have not been yet listed
    stock_data = stock_data.loc[:, (stock_data != 0).any(axis=0)]  # drop the stocks that were delisted

    # Calculate TE for each calculation methods
    #methods = ['kernel', 'mi', 'granger']
    methods = ['correlation']
    for method in methods:
        # Get combinations of every stock ids in the file
        comb_list = [combs for combs in combinations(stock_data.columns, 2)]
        print('total combinations are: ' + str(len(comb_list)))
        print('=' * 20 + ' ' * 5 + method + ' ' * 5 + '=' * 20)
        calculation = []
        for comb_no in tqdm(range(len(comb_list)), unit='combination', desc='calculating'):
            X = comb_list[comb_no][0]
            Y = comb_list[comb_no][1]
            reduced_data = stock_data[[Y, X]].dropna().values
            if method == 'kernel':
                result_list = kde(reduced_data, Y, X)
            elif method == 'mi':
                result_list = mi(reduced_data, Y, X)
            elif method == 'granger':
                result_list = granger(reduced_data, Y, X)
            else:
                result_list = corr(reduced_data, Y, X)
            calculation = calculation + result_list  # add the lists into a larger list
        if method == 'mi':
            result_df = pd.DataFrame(calculation, columns=['exog', 'endog', 'MI_XY', 'Norm_MI_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_MI_XY'])
        elif method == 'correlation':
            result_df = pd.DataFrame(calculation, columns=['exog', 'endog', 'Correlation', 'p_value_XY'])
        else:
            result_df = pd.DataFrame(calculation, columns=['exog', 'endog', 'TE_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_TE_XY'])
            result_df['ETE_XY'] = result_df['TE_XY'] - result_df['Ave_TE_XY']  # Calculate effective TE
        print(result_df.head())
        result_df.to_csv('Data/krx350_prior_half_{}/{}_{}_TE.csv'.format(method, files, method))
