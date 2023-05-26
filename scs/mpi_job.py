from mpi4py import MPI
import pandas as pd
from itertools import combinations
from Entropy_Causality import TransferEntropyArray
from Granger_Causality import GrangerCausality
from lag_data_gen import LaggedTimeSeriesDF

def transfer_ent(lts_array, endog, exog, t):
    """
    Calculate TE using KDE
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropyArray(lts_array, endog=endog, exog=exog, lag=t)
    te_res = te.TransferEntropy(n_shuffles=20)
    result = te_res.transpose().tolist()

    result[0] = [endog, exog] + result[0]
    result[1] = [exog, endog] + result[1]
    return result

def granger(lts_array, endog, exog, t):
    """
    Calculate TE using granger causality
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    gc = GrangerCausality(lts_array, endog=endog, exog=exog, lag=t)
    gc_res = gc.Granger_Caus(n_shuffles=20)
    result = gc_res.transpose().tolist()

    result[0] = [endog, exog] + result[0]
    result[1] = [exog, endog] + result[1]
    return result

def mutual_inf(lts_array, endog, exog, t):
    """
    Calculate Mutual Information using KDE
    Creates calculation results of MI into a list as [X, Y, MI_XY, p_value_XY, z_score_XY, Ave_MI_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    mi = TransferEntropyArray(lts_array, endog=endog, exog=exog, lag=t, method='mi')
    mi_res = mi.MutualInformation(n_shuffles=20)
    in_result = mi_res.transpose().tolist()

    result = [[endog, exog] + in_result[0]]
    return result

def mpi_work(data_name, data, comb_list, lag, method):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cals = len(comb_list)
    N = int(cals / size)

    if rank == 0:
        combination_lst = comb_list
    else:
        combination_lst = None

    comb_sub = comm.scatter(combination_lst, root=0)
    
    calculation = []

    for comb_no in range(N):
        comb = comb_sub[comb_no]
        X = comb[0]
        Y = comb[1]
        reduced_data = data[[Y, X]]
        lts_array = LaggedTimeSeriesDF(reduced_data, lag=lag).df.to_numpy()
        
        if method == 'transfer_ent':
            result_list = transfer_ent(lts_array, X, Y, lag)
        elif method == 'granger':
            result_list = granger(lts_array, X, Y, lag)
        else:
            result_list = mutual_inf(lts_array, X, Y, lag)

        calculation = calculation + result_list
    
    res = comm.gather(calculation, root=0)

    if rank == 0:
        if method == 'transfer_ent':
            result_df = pd.DataFrame(res, columns=['endog', 'exog', 'TE_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_TE_XY'])
            result_df['ETE_XY'] = result_df['TE_XY'] - result_df['Ave_TE_XY']
        elif method == 'granger':
            result_df = pd.DataFrame(res, columns=['endog', 'exog', 'GC_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_GC_XY'])
        else:
            result_df = pd.DataFrame(res, columns=['endog', 'exog', 'Norm_MI_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_MI_XY'])

        result_df.to_csv(f'./data/{method}_{data_name}_mpi_result.csv', index=False)
    
    return

if __name__ == "__main__":
    data_name = 'test_data'
    data = pd.read_csv(f'./data/{data_name}.csv')
    comb_list = [combs for combs in combinations(data.columns, 2)]
    lag = 1
    method = 'granger'
    mpi_work(data_name, data, comb_list, lag, method)



