import pandas as pd
import numpy as np
from PyCausality.TransferEntropy import *


def kde(DataFrame, Y, X):
    TE = TransferEntropy(DF=DataFrame,
                         endog=Y,  # Dependent Variable
                         exog=X,  # Independent Variable
                         lag=2)
    TE.nonlinear_TE(pdf_estimator='kernel', n_shuffles=100)
    print(TE.results)
    result_list = TE.results.values.reshape((4, 2)).transpose().tolist()
    result_list[0] = [X, Y] + result_list[0]
    result_list[1] = [Y, X] + result_list[1]
    return result_list


def histogram(DataFrame, Y, X):
    TE = TransferEntropy(DF=DataFrame,
                         endog=Y,  # Dependent Variable
                         exog=X,  # Independent Variable
                         lag=2)
    TE.nonlinear_TE(pdf_estimator='histogram', n_shuffles=100)
    print(TE.results)
    result_list = TE.results.values.reshape((4, 2)).transpose().tolist()
    result_list[0] = [X, Y] + result_list[0]
    result_list[1] = [Y, X] + result_list[1]
    return result_list


def granger(DataFrame, Y, X):
    TE = TransferEntropy(DF=DataFrame,
                         endog=Y,  # Dependent Variable
                         exog=X,  # Independent Variable
                         lag=2)
    TE.linear_TE(n_shuffles=100)
    print(TE.results)
    result_list = TE.results.values.reshape((4, 2)).transpose().tolist()
    result_list[0] = [X, Y] + result_list[0]
    result_list[1] = [Y, X] + result_list[1]
    return result_list


rand = np.random.RandomState(seed=23)
dt_indx = pd.date_range(start="2018-1-1", periods=100)
DataFrame = pd.DataFrame(rand.randn(100, 2), columns={'MSTF', 'AAPL'}, index=dt_indx)
# print(DataFrame)
X = DataFrame.columns[1]
Y = DataFrame.columns[0]
#TE = TransferEntropy(   DF = DataFrame,
#                        endog = 'MSTF',     # Dependent Variable
#                        exog = 'AAPL',      # Independent Variable
#                        lag = 2)

# print(TE)
print('='*20, 'Granger Causality', '='*20)
print(granger(DataFrame, Y, X))
print('='*20, 'Kernel', '='*20)
print(kde(DataFrame, Y, X))
#print('='*20, 'Histogram', '='*20)
#print(histogram(DataFrame, Y, X))
